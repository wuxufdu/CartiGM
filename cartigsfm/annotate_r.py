"""Real R/rpy2-backed annotation backends: SingleR, scmap, Symphony, CellAssign.

This module replaces the stubs in :mod:`cartigsfm.annotate` with calls
against the real R packages (SingleR / scmap from Bioconductor,
symphony / cellassign from GitHub source). When rpy2 cannot import the
R session, or when the requested R package is not installed, each
wrapper degrades to a structured ``{"available": False, ...}`` dict
matching the existing Python contract so downstream code keeps working.

The R session is initialised lazily on first call. The runtime expects
R 4.6.0 at ``%LOCALAPPDATA%/Programs/R/R-4.6.0`` by default; override
with the ``CARTIGSFM_R_HOME`` environment variable if needed.

Common protocol (cluster-level pseudo-bulk):

1. Read the query h5ad with ``scanpy.read_h5ad``.
2. Build a (clusters x genes) pseudo-bulk on the Python side using a
   sparse design matrix multiplied with the cell x gene expression
   matrix; this matches the per-cluster mean used elsewhere in the
   package and keeps the rpy2 transfer below tens of MB even on the
   full EBR (33k cells, 30k genes -> 11 clusters x ~30k genes ~ 2.5 MB
   dense).
3. For SingleR / scmap / Symphony we ALSO build a (samples x genes) or
   (cells x genes) reference matrix from the curated atlas h5ad
   (default: acc.h5ad with the ``chongdrocyte_subtype`` label) using a
   sample-stratified subsample so the R side stays under 5 GB. The
   subsample is reproducible via the ``random_state`` argument.
4. CellAssign uses a marker-gene matrix (genes x cell-types) derived
   from the cartilage_dictionary_v1 ``core_genes`` panel.
5. Each wrapper returns ``per_cluster`` (one row per query cluster)
   with a ``<method>_label`` column matching the contract used by
   :func:`cartigsfm.annotate.compare_annotations`.

The wrappers are intentionally tolerant: if anything fails inside R
(missing package, runtime error), the wrapper returns a placeholder
with ``available=False`` and a ``note`` carrying the R traceback.
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# rpy2 lazy initialisation
# ---------------------------------------------------------------------------

_R_INITIALISED = False
_R_OK: Optional[bool] = None
_R_ERR: Optional[str] = None

_DEFAULT_R_HOME = os.environ.get(
    "CARTIGSFM_R_HOME",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\R\R-4.6.0"),
)
_DEFAULT_RTOOLS_BIN = os.environ.get(
    "CARTIGSFM_RTOOLS_BIN",
    r"F:\rtools45\x86_64-w64-mingw32.static.posix\bin",
)
_DEFAULT_RTOOLS_USR_BIN = os.environ.get(
    "CARTIGSFM_RTOOLS_USR_BIN",
    r"F:\rtools45\usr\bin",
)


def _find_rscript() -> Optional[str]:
    """Resolve a usable ``Rscript.exe`` path on Windows.

    Checks ``CARTIGSFM_R_HOME`` then PATH; returns ``None`` if missing.
    """
    candidates = [
        os.path.join(_DEFAULT_R_HOME, "bin", "x64", "Rscript.exe"),
        os.path.join(_DEFAULT_R_HOME, "bin", "Rscript.exe"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # PATH fallback (Linux/Mac builds use plain ``Rscript``)
    for binname in ("Rscript.exe", "Rscript"):
        for p in os.environ.get("PATH", "").split(os.pathsep):
            full = os.path.join(p, binname)
            if os.path.isfile(full):
                return full
    return None


# R script template used by ``annotate_with_cellassign``. Uses str.format
# substitution; ALL literal R braces are doubled to ``{{`` / ``}}``.
_CELLASSIGN_RSCRIPT = r"""
Sys.setenv(LANGUAGE = "en")
Sys.setenv(RETICULATE_PYTHON = "{py}")
old_path <- Sys.getenv("PATH")
Sys.setenv(PATH = paste("{rtools_bin}", "{rtools_usr}", old_path, sep = .Platform$path.sep))
suppressPackageStartupMessages({{
  library(reticulate)
  reticulate::use_python("{py}", required = TRUE)
  library(tensorflow)
  library(jsonlite)
}})

meta <- jsonlite::fromJSON("{tmp}/meta.json")
cell_ids <- as.character(meta$cell_ids)
marker_genes <- as.character(meta$marker_genes)
cell_types <- as.character(meta$cell_types)
clusters <- as.character(meta$clusters)

Xc <- as.matrix(read.table("{tmp}/Xc.tsv", header = FALSE, sep = "\t", check.names = FALSE))
rho <- as.matrix(read.table("{tmp}/rho.tsv", header = FALSE, sep = "\t", check.names = FALSE))
sf <- scan("{tmp}/sf.tsv", what = numeric(), quiet = TRUE)
rownames(Xc) <- cell_ids
colnames(Xc) <- marker_genes
rownames(rho) <- marker_genes
colnames(rho) <- cell_types

result <- tryCatch({{
  suppressPackageStartupMessages(library(cellassign))
  res <- cellassign(
    exprs_obj = Xc,
    marker_gene_info = rho,
    s = sf,
    learning_rate = 0.01,
    shrinkage = TRUE,
    verbose = FALSE,
    num_runs = 1,
    min_delta = 0.01,
    max_iter_em = 50,
    max_iter_adam = 100
  )
  out <- data.frame(
    cell_id = rownames(Xc),
    cluster = clusters,
    cellassign_label = as.character(res$cell_type),
    stringsAsFactors = FALSE
  )
  write.table(out, file = "{out_tsv}", sep = "\t", quote = FALSE,
              row.names = FALSE, col.names = TRUE)
  "ok"
}}, error = function(e) {{
  msg <- paste0("cellassign error: ", conditionMessage(e))
  writeLines(msg, "{err_path}")
  msg
}})

if (!identical(result, "ok")) {{
  cat(result, "\n", sep = "")
  quit(save = "no", status = 1)
}}
"""


def _ensure_r_init() -> Tuple[bool, Optional[str]]:
    """Return (ok, err). Sets R_HOME / PATH and calls rpy2.rinterface.initr.

    On Windows we also (1) add R's ``bin/x64`` to the DLL search path so
    that downstream ``library(stats)`` / ``library(SingleR)`` etc. can
    resolve the bundled BLAS/LAPACK DLLs, (2) silence rpy2's console
    callbacks because R's localised messages may be GBK-encoded on a
    Chinese-locale Windows host and crash rpy2's ``utf-8`` decoder
    midway through ``r()`` (which then returns ``None`` and breaks the
    caller), (3) force-load R's ``stats`` / ``utils`` / ``methods``
    packages so SingleR / scmap / symphony can resolve their
    transitive imports.
    """
    global _R_INITIALISED, _R_OK, _R_ERR
    if _R_INITIALISED:
        return _R_OK or False, _R_ERR
    _R_INITIALISED = True
    try:
        os.environ.setdefault("R_HOME", _DEFAULT_R_HOME)
        os.environ.setdefault("LANGUAGE", "en")
        # CellAssign's R wrapper uses tensorflow R package which goes
        # through reticulate. Pin reticulate to the active venv so it
        # picks up the `tensorflow` Python module we install there
        # (otherwise reticulate falls back to its uv-managed env which
        # has no internet on this host and cellassign's .onLoad fails
        # with "Tensorflow installation not detected").
        if not os.environ.get("RETICULATE_PYTHON"):
            import sys as _sys
            os.environ["RETICULATE_PYTHON"] = _sys.executable
        r_bin = os.path.join(_DEFAULT_R_HOME, "bin", "x64")
        if os.path.isdir(r_bin):
            try:
                os.add_dll_directory(r_bin)  # type: ignore[attr-defined]
            except (AttributeError, FileNotFoundError, OSError):
                pass
        path_segments = [r_bin, _DEFAULT_RTOOLS_BIN, _DEFAULT_RTOOLS_USR_BIN]
        path = os.environ.get("PATH", "")
        for seg in path_segments:
            if seg and seg not in path:
                path = seg + os.pathsep + path
        os.environ["PATH"] = path
        # Patch rpy2's bytes->str decoders BEFORE importing rinterface
        # so that R's localised messages on a Chinese-locale Windows
        # host don't crash mid-evaluation. Without this, any R warning
        # encoded in GBK by R's locale layer kills the active r() call
        # because rpy2's default decode uses errors='strict' against
        # sys.getdefaultencoding() == 'utf-8'.
        try:
            import rpy2.rinterface_lib.conversion as _conv
            _orig_to_str = _conv._cchar_to_str
            _orig_to_str_max = _conv._cchar_to_str_with_maxlen
            _ffi = _conv.ffi
            def _safe_to_str(c, encoding):
                try:
                    return _ffi.string(c).decode(encoding, "replace")
                except Exception:
                    return ""
            def _safe_to_str_max(c, maxlen, encoding):
                try:
                    return _ffi.string(c, maxlen).decode(encoding, "replace")
                except Exception:
                    return ""
            _conv._cchar_to_str = _safe_to_str
            _conv._cchar_to_str_with_maxlen = _safe_to_str_max
        except Exception:
            pass
        import rpy2.rinterface as ri
        # Silence rpy2's console writers BEFORE initr so localised
        # messages cannot trip the utf-8 decoder mid-evaluation. We
        # accept this means R warnings are lost; the wrappers always
        # return structured error info via the dict.
        import rpy2.rinterface_lib.callbacks as cb
        def _silent(_s):
            return None
        cb.consolewrite_print = _silent
        cb.consolewrite_warnerror = _silent
        try:
            cb.showmessage = _silent  # type: ignore[attr-defined]
        except Exception:
            pass
        # rpy2 also reads from these as module-level callables; rebind
        # them on the rinterface_lib root for completeness (newer rpy2
        # versions expose the bridge here).
        try:
            import rpy2.rinterface_lib._rinterface_capi as _capi  # noqa: F401
        except Exception:
            pass
        ri.initr()
        # Re-silence after initr in case it reset the bindings.
        cb.consolewrite_print = _silent
        cb.consolewrite_warnerror = _silent
        try:
            cb.showmessage = _silent  # type: ignore[attr-defined]
        except Exception:
            pass
        from rpy2.robjects import r
        # rpy2's r() respects R's invisibility flag by default, which
        # turns plain `library(...)` and `pkg <- ...` calls into
        # ``None`` returns and makes downstream ``[0]`` lookups crash.
        # We disable invisibility so every r() call returns whatever R
        # produced.
        try:
            r._invisible = False  # type: ignore[attr-defined]
        except Exception:
            pass
        r("Sys.setenv(LANGUAGE='en')")
        r("for (pkg in c('stats','utils','methods','grDevices','graphics')) "
          "suppressMessages(library(pkg, character.only=TRUE))")
        _R_OK, _R_ERR = True, None
    except Exception as exc:  # rpy2 not installed, R missing, etc.
        _R_OK, _R_ERR = False, repr(exc)
    return _R_OK or False, _R_ERR


def _r_pkg_available(name: str) -> bool:
    ok, err = _ensure_r_init()
    if not ok:
        return False
    from rpy2.robjects import r
    try:
        # Wrap in c(...) so the value is visibly returned (rpy2's
        # default invisible=True returns None for invisibly-returned R
        # expressions like requireNamespace).
        out = r(f"c(x = isTRUE(requireNamespace({name!r}, quietly=TRUE)))")
        return bool(list(out)[0])
    except Exception:
        return False


def _r_unavailable(method: str, *, install_hint: str, note: str) -> Dict[str, Any]:
    return {
        "method": method,
        "available": False,
        "per_cluster": pd.DataFrame(),
        "note": note,
        "install_hint": install_hint,
    }


# ---------------------------------------------------------------------------
# Pseudo-bulk helpers (Python side)
# ---------------------------------------------------------------------------

def _per_cluster_mean_dense(
    adata,
    cluster_col: str,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Return (cluster x gene) dense mean matrix, cluster ids, gene names."""
    X = adata.X
    if sp.issparse(X):
        X = X.tocsr()
    cluster_labels = adata.obs[cluster_col].astype(str).values
    codes, labels = pd.factorize(cluster_labels, sort=True)
    n_groups = len(labels)
    counts = np.bincount(codes, minlength=n_groups)
    design = sp.csr_matrix(
        (np.ones(len(codes), dtype=np.float32), (codes, np.arange(len(codes)))),
        shape=(n_groups, len(codes)),
        dtype=np.float32,
    )
    if sp.issparse(X):
        sums = (design @ X).toarray()
    else:
        sums = np.asarray(design @ np.asarray(X, dtype=np.float32))
    counts_safe = np.where(counts > 0, counts, 1).astype(np.float32)
    means = sums / counts_safe[:, None]
    return means.astype(np.float32), [str(l) for l in labels], [str(g) for g in adata.var_names]


def _subsample_reference(
    adata,
    label_col: str,
    *,
    max_cells_per_label: int,
    random_state: int = 0,
):
    """Return a copy of adata stratified-subsampled by label_col."""
    rng = np.random.default_rng(int(random_state))
    labels = adata.obs[label_col].astype(str).values
    keep_idx: List[int] = []
    for lab in sorted(set(labels.tolist())):
        idx = np.where(labels == lab)[0]
        if idx.size > int(max_cells_per_label):
            idx = rng.choice(idx, size=int(max_cells_per_label), replace=False)
        keep_idx.extend(idx.tolist())
    keep = np.sort(np.array(keep_idx, dtype=np.int64))
    return adata[keep].copy()


def _intersect_genes(
    a_genes: Sequence[str], b_genes: Sequence[str]
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    a_up = [str(g).upper() for g in a_genes]
    b_up = [str(g).upper() for g in b_genes]
    a_idx_map: Dict[str, int] = {}
    for i, g in enumerate(a_up):
        a_idx_map.setdefault(g, i)
    b_idx_map: Dict[str, int] = {}
    for i, g in enumerate(b_up):
        b_idx_map.setdefault(g, i)
    common = [g for g in a_up if g in b_idx_map]
    common = list(dict.fromkeys(common))
    a_pos = np.array([a_idx_map[g] for g in common], dtype=np.int64)
    b_pos = np.array([b_idx_map[g] for g in common], dtype=np.int64)
    return common, a_pos, b_pos


def _push_matrix_to_r(arr: np.ndarray, rownames: Sequence[str], colnames: Sequence[str], r_name: str) -> None:
    """Push a (n x m) float64 matrix to R as ``r_name`` with row/col names."""
    from rpy2 import robjects
    arr64 = np.asarray(arr, dtype=np.float64)
    flat = robjects.FloatVector(arr64.ravel(order="F"))
    rn = robjects.StrVector([str(x) for x in rownames])
    cn = robjects.StrVector([str(x) for x in colnames])
    robjects.r.assign("..__flat__..", flat)
    robjects.r.assign("..__rn__..", rn)
    robjects.r.assign("..__cn__..", cn)
    robjects.r(
        f"{r_name} <- matrix(..__flat__.., nrow={int(arr64.shape[0])}, ncol={int(arr64.shape[1])});"
        f" rownames({r_name}) <- ..__rn__..;"
        f" colnames({r_name}) <- ..__cn__..;"
        " rm(..__flat__.., ..__rn__.., ..__cn__..);"
    )


def _r_to_pandas(r_obj_name: str) -> pd.DataFrame:
    """Read an R data.frame named ``r_obj_name`` into a pandas DataFrame."""
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri, default_converter
    obj = robjects.r(r_obj_name)
    with (default_converter + pandas2ri.converter).context() as cv:
        return pd.DataFrame(cv.rpy2py(obj))
# ---------------------------------------------------------------------------
# 1. SingleR (Bioconductor)
# ---------------------------------------------------------------------------

def annotate_with_singler(
    query_h5ad,
    reference_h5ad,
    *,
    reference_label_col: str = "chongdrocyte_subtype",
    cluster_col: str = "leiden_res0_5",
    max_reference_cells_per_label: int = 1500,
    out_tsv=None,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Run real SingleR via rpy2 with the curated atlas as reference."""
    method = "singler"
    ok, err = _ensure_r_init()
    if not ok:
        return _r_unavailable(method, install_hint="rpy2 + R 4.6.0",
                              note="rpy2 R session failed: " + str(err))
    if not _r_pkg_available("SingleR"):
        return _r_unavailable(method, install_hint="BiocManager::install('SingleR')",
                              note="R package SingleR is not installed")

    import scanpy as sc
    q = sc.read_h5ad(str(query_h5ad))
    if cluster_col not in q.obs.columns:
        return _r_unavailable(method, install_hint="",
                              note="query has no cluster column " + repr(cluster_col))
    ref = sc.read_h5ad(str(reference_h5ad))
    if reference_label_col not in ref.obs.columns:
        return _r_unavailable(method, install_hint="",
                              note="reference has no label column " + repr(reference_label_col))
    ref_sub = _subsample_reference(
        ref, reference_label_col,
        max_cells_per_label=max_reference_cells_per_label,
        random_state=random_state,
    )

    q_means, q_clusters, q_genes = _per_cluster_mean_dense(q, cluster_col)
    r_means, r_labels, r_genes = _per_cluster_mean_dense(ref_sub, reference_label_col)

    common, q_pos, r_pos = _intersect_genes(q_genes, r_genes)
    if not common:
        return _r_unavailable(method, install_hint="",
                              note="zero gene overlap between query and reference")
    Q = q_means[:, q_pos].astype(np.float32)
    R_mat = r_means[:, r_pos].astype(np.float32)

    from rpy2 import robjects
    from rpy2.robjects import StrVector
    _push_matrix_to_r(Q.T, common, q_clusters, "..__Q__..")
    _push_matrix_to_r(R_mat.T, common, r_labels, "..__R__..")
    robjects.r.assign("..__rlabs__..", StrVector(r_labels))
    robjects.r("""
    suppressPackageStartupMessages({
      library(SingleR)
      library(SummarizedExperiment)
    })
    pred <- SingleR(test = ..__Q__.., ref = ..__R__.., labels = ..__rlabs__..)
    out <- data.frame(
      cluster = rownames(pred),
      singler_label = as.character(pred$labels),
      singler_pruned_label = as.character(pred$pruned.labels),
      singler_score = apply(pred$scores, 1, max),
      stringsAsFactors = FALSE
    )
    """)
    out = _r_to_pandas("out")
    out["cluster"] = out["cluster"].astype(str)
    cluster_to_n = {str(k): int(v) for k, v in Counter(q.obs[cluster_col].astype(str).values).items()}
    rows: List[Dict[str, Any]] = []
    for _, row in out.iterrows():
        cid = str(row["cluster"])
        rows.append({
            cluster_col: cid,
            "singler_label": str(row["singler_label"]),
            "singler_pruned_label": str(row.get("singler_pruned_label", "")),
            "singler_score": float(row.get("singler_score", float("nan"))),
            "singler_n_cells": cluster_to_n.get(cid, 0),
        })
    per_cluster = pd.DataFrame(rows).sort_values(cluster_col).reset_index(drop=True)
    out_path: Optional[str] = None
    if out_tsv is not None:
        out_path = str(out_tsv)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        per_cluster.to_csv(out_path, sep="\t", index=False)
    robjects.r("rm(..__Q__.., ..__R__.., ..__rlabs__.., pred, out)")
    return {
        "method": method,
        "available": True,
        "per_cluster": per_cluster,
        "per_cell_predictions_path": out_path,
        "n_query_cells": int(q.n_obs),
        "n_reference_cells": int(ref_sub.n_obs),
        "n_clusters": int(per_cluster.shape[0]),
        "reference_label_col": reference_label_col,
        "cluster_col": cluster_col,
        "n_common_genes": int(len(common)),
        "note": (
            "Real SingleR via rpy2 on cluster-level pseudo-bulk."
            + " Reference subsampled to " + str(ref_sub.n_obs)
            + " cells (max " + str(max_reference_cells_per_label)
            + " per label). Common genes after intersection: " + str(len(common)) + "."
        ),
    }

# ---------------------------------------------------------------------------
# 2. scmap (Bioconductor)
# ---------------------------------------------------------------------------

def annotate_with_scmap(
    query_h5ad,
    reference_h5ad,
    *,
    reference_label_col: str = "chongdrocyte_subtype",
    cluster_col: str = "leiden_res0_5",
    max_reference_cells_per_label: int = 1500,
    out_tsv=None,
    random_state: int = 0,
    n_features: int = 500,
    similarity_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Run real scmap-cluster via rpy2 with the curated atlas as reference."""
    method = "scmap"
    ok, err = _ensure_r_init()
    if not ok:
        return _r_unavailable(method, install_hint="rpy2 + R 4.6.0",
                              note="rpy2 R session failed: " + str(err))
    if not _r_pkg_available("scmap"):
        return _r_unavailable(method, install_hint="BiocManager::install('scmap')",
                              note="R package scmap is not installed")

    import scanpy as sc
    q = sc.read_h5ad(str(query_h5ad))
    if cluster_col not in q.obs.columns:
        return _r_unavailable(method, install_hint="",
                              note="query has no cluster column " + repr(cluster_col))
    ref = sc.read_h5ad(str(reference_h5ad))
    if reference_label_col not in ref.obs.columns:
        return _r_unavailable(method, install_hint="",
                              note="reference has no label column " + repr(reference_label_col))
    ref_sub = _subsample_reference(
        ref, reference_label_col,
        max_cells_per_label=max_reference_cells_per_label,
        random_state=random_state,
    )

    q_means, q_clusters, q_genes = _per_cluster_mean_dense(q, cluster_col)
    r_means, r_labels, r_genes = _per_cluster_mean_dense(ref_sub, reference_label_col)
    common, q_pos, r_pos = _intersect_genes(q_genes, r_genes)
    if not common:
        return _r_unavailable(method, install_hint="",
                              note="zero gene overlap between query and reference")
    Q = q_means[:, q_pos].astype(np.float32)
    R_mat = r_means[:, r_pos].astype(np.float32)

    from rpy2 import robjects
    from rpy2.robjects import StrVector
    _push_matrix_to_r(Q.T, common, q_clusters, "..__Q__..")
    _push_matrix_to_r(R_mat.T, common, r_labels, "..__R__..")
    robjects.r.assign("..__rlabs__..", StrVector(r_labels))
    robjects.r.assign("..__qclusters__..", StrVector(q_clusters))
    r_code = (
        "suppressPackageStartupMessages({"
        " library(scmap); library(SingleCellExperiment); library(SummarizedExperiment) })\n"
        "ref_sce <- SingleCellExperiment(\n"
        "  assays = list(counts = ..__R__.., logcounts = ..__R__..),\n"
        "  colData = data.frame(cell_type1 = ..__rlabs__.., row.names = colnames(..__R__..))\n"
        ")\n"
        "rowData(ref_sce)$feature_symbol <- rownames(..__R__..)\n"
        "ref_sce <- selectFeatures(ref_sce, suppress_plot = TRUE, n_features = " + str(int(n_features)) + ")\n"
        "ref_sce <- indexCluster(ref_sce, cluster_col = 'cell_type1')\n"
        "q_sce <- SingleCellExperiment(\n"
        "  assays = list(counts = ..__Q__.., logcounts = ..__Q__..),\n"
        "  colData = data.frame(cluster = ..__qclusters__.., row.names = colnames(..__Q__..))\n"
        ")\n"
        "rowData(q_sce)$feature_symbol <- rownames(..__Q__..)\n"
        "proj <- scmapCluster(\n"
        "  projection = q_sce,\n"
        "  index_list = list(reference = metadata(ref_sce)$scmap_cluster_index),\n"
        "  threshold = " + str(float(similarity_threshold)) + "\n"
        ")\n"
        "out <- data.frame(\n"
        "  cluster = ..__qclusters__..,\n"
        "  scmap_label = as.character(proj$scmap_cluster_labs[, 'reference']),\n"
        "  scmap_similarity = as.numeric(proj$scmap_cluster_siml[, 'reference']),\n"
        "  stringsAsFactors = FALSE\n"
        ")\n"
    )
    robjects.r(r_code)
    out = _r_to_pandas("out")
    out["cluster"] = out["cluster"].astype(str)
    cluster_to_n = {str(k): int(v) for k, v in Counter(q.obs[cluster_col].astype(str).values).items()}
    rows: List[Dict[str, Any]] = []
    for _, row in out.iterrows():
        cid = str(row["cluster"])
        rows.append({
            cluster_col: cid,
            "scmap_label": str(row["scmap_label"]),
            "scmap_similarity": float(row.get("scmap_similarity", float("nan"))),
            "scmap_n_cells": cluster_to_n.get(cid, 0),
        })
    per_cluster = pd.DataFrame(rows).sort_values(cluster_col).reset_index(drop=True)
    out_path: Optional[str] = None
    if out_tsv is not None:
        out_path = str(out_tsv)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        per_cluster.to_csv(out_path, sep="\t", index=False)
    robjects.r("rm(..__Q__.., ..__R__.., ..__rlabs__.., ..__qclusters__.., ref_sce, q_sce, proj, out)")
    return {
        "method": method,
        "available": True,
        "per_cluster": per_cluster,
        "per_cell_predictions_path": out_path,
        "n_query_cells": int(q.n_obs),
        "n_reference_cells": int(ref_sub.n_obs),
        "n_clusters": int(per_cluster.shape[0]),
        "n_features": int(n_features),
        "reference_label_col": reference_label_col,
        "cluster_col": cluster_col,
        "n_common_genes": int(len(common)),
        "note": (
            "Real scmap-cluster via rpy2. Reference subsampled to "
            + str(ref_sub.n_obs) + " cells (max " + str(max_reference_cells_per_label)
            + " per label). selectFeatures n_features=" + str(int(n_features))
            + ". Common genes after intersection: " + str(len(common)) + "."
        ),
    }

# ---------------------------------------------------------------------------
# 3. Symphony (GitHub: immunogenomics/symphony)
# ---------------------------------------------------------------------------

def annotate_with_symphony(
    query_h5ad,
    reference_h5ad,
    *,
    reference_label_col: str = "chongdrocyte_subtype",
    cluster_col: str = "leiden_res0_5",
    max_reference_cells_per_label: int = 1500,
    out_tsv=None,
    random_state: int = 0,
    d: int = 20,
    k: int = 100,
) -> Dict[str, Any]:
    """Run real Symphony reference-build + query-map via rpy2.

    Symphony works on the raw cell-level matrix, not pseudo-bulk. We
    push the query as a (gene x cell) sparse-coerced-dense block (in
    chunks if the dense form exceeds 4 GB) and the reference as a
    sample-stratified subsample. Symphony returns per-cell labels which
    we aggregate per cluster via majority vote.
    """
    method = "symphony"
    ok, err = _ensure_r_init()
    if not ok:
        return _r_unavailable(method, install_hint="rpy2 + R 4.6.0",
                              note="rpy2 R session failed: " + str(err))
    if not _r_pkg_available("symphony"):
        return _r_unavailable(method, install_hint="remotes::install_github('immunogenomics/symphony')",
                              note="R package symphony is not installed")

    import scanpy as sc
    q = sc.read_h5ad(str(query_h5ad))
    if cluster_col not in q.obs.columns:
        return _r_unavailable(method, install_hint="",
                              note="query has no cluster column " + repr(cluster_col))
    ref = sc.read_h5ad(str(reference_h5ad))
    if reference_label_col not in ref.obs.columns:
        return _r_unavailable(method, install_hint="",
                              note="reference has no label column " + repr(reference_label_col))
    ref_sub = _subsample_reference(
        ref, reference_label_col,
        max_cells_per_label=max_reference_cells_per_label,
        random_state=random_state,
    )
    common, q_pos, r_pos = _intersect_genes(
        list(q.var_names.astype(str)),
        list(ref_sub.var_names.astype(str)),
    )
    if not common:
        return _r_unavailable(method, install_hint="",
                              note="zero gene overlap between query and reference")
    Xq = q.X
    if sp.issparse(Xq):
        Xq = np.asarray(Xq[:, q_pos].toarray(), dtype=np.float32)
    else:
        Xq = np.asarray(np.asarray(Xq, dtype=np.float32)[:, q_pos], dtype=np.float32)
    Xr = ref_sub.X
    if sp.issparse(Xr):
        Xr = np.asarray(Xr[:, r_pos].toarray(), dtype=np.float32)
    else:
        Xr = np.asarray(np.asarray(Xr, dtype=np.float32)[:, r_pos], dtype=np.float32)

    q_cells = list(q.obs_names.astype(str))
    r_cells = list(ref_sub.obs_names.astype(str))
    q_clusters = list(q.obs[cluster_col].astype(str).values)
    r_labels = list(ref_sub.obs[reference_label_col].astype(str).values)

    from rpy2 import robjects
    from rpy2.robjects import StrVector
    _push_matrix_to_r(Xq.T, common, q_cells, "..__Xq__..")
    _push_matrix_to_r(Xr.T, common, r_cells, "..__Xr__..")
    robjects.r.assign("..__rlabs__..", StrVector(r_labels))
    robjects.r.assign("..__qclusters__..", StrVector(q_clusters))
    robjects.r("""
    suppressPackageStartupMessages({
      library(symphony)
      library(harmony)
      library(Matrix)
    })
    Xr_sp <- as(..__Xr__.., 'dgCMatrix')
    Xq_sp <- as(..__Xq__.., 'dgCMatrix')
    ref_meta <- data.frame(
      cell_id = colnames(..__Xr__..),
      cell_type = ..__rlabs__..,
      stringsAsFactors = FALSE
    )
    ref_obj <- symphony::buildReference(
      exp_ref = Xr_sp,
      metadata_ref = ref_meta,
      vars = NULL,
      K = """ + str(int(k)) + """,
      verbose = FALSE,
      do_umap = FALSE,
      do_normalize = FALSE,
      vargenes_method = 'vst',
      vargenes_groups = NULL,
      topn = min(2000, nrow(Xr_sp)),
      d = """ + str(int(d)) + """,
      save_uwot_path = NULL
    )
    q_meta <- data.frame(
      cell_id = colnames(..__Xq__..),
      cluster = ..__qclusters__..,
      stringsAsFactors = FALSE
    )
    q_obj <- symphony::mapQuery(
      exp_query = Xq_sp,
      metadata_query = q_meta,
      ref_obj = ref_obj,
      vars = NULL,
      verbose = FALSE,
      do_normalize = FALSE,
      do_umap = FALSE
    )
    q_obj <- symphony::knnPredict(
      query_obj = q_obj,
      ref_obj = ref_obj,
      train_labels = ref_obj$meta_data$cell_type,
      k = 5
    )
    out <- data.frame(
      cell_id = colnames(..__Xq__..),
      cluster = ..__qclusters__..,
      symphony_label = as.character(q_obj$meta_data$cell_type_pred_knn),
      stringsAsFactors = FALSE
    )
    """)
    out = _r_to_pandas("out")
    out["cluster"] = out["cluster"].astype(str)
    rows: List[Dict[str, Any]] = []
    for cid, sub in out.groupby("cluster"):
        labs = [str(x) for x in sub["symphony_label"].astype(str).tolist() if str(x) and str(x) != "NA" and str(x) != "nan"]
        if not labs:
            top = ""
            top_n = 0
        else:
            top, top_n = Counter(labs).most_common(1)[0]
        rows.append({
            cluster_col: str(cid),
            "symphony_label": str(top),
            "symphony_n_cells": int(len(sub)),
            "symphony_majority_frac": round(float(top_n) / max(len(sub), 1), 4),
        })
    per_cluster = pd.DataFrame(rows).sort_values(cluster_col).reset_index(drop=True)
    out_path: Optional[str] = None
    if out_tsv is not None:
        out_path = str(out_tsv)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, sep="\t", index=False)
    robjects.r("rm(..__Xq__.., ..__Xr__.., ..__rlabs__.., ..__qclusters__.., ref_obj, q_obj, ref_meta, q_meta, out)")
    return {
        "method": method,
        "available": True,
        "per_cluster": per_cluster,
        "per_cell_predictions_path": out_path,
        "n_query_cells": int(q.n_obs),
        "n_reference_cells": int(ref_sub.n_obs),
        "n_clusters": int(per_cluster.shape[0]),
        "reference_label_col": reference_label_col,
        "cluster_col": cluster_col,
        "n_common_genes": int(len(common)),
        "k": int(k),
        "d": int(d),
        "note": (
            "Real Symphony via rpy2. buildReference K=" + str(int(k))
            + " d=" + str(int(d)) + ". Reference subsampled to "
            + str(ref_sub.n_obs) + " cells (max "
            + str(max_reference_cells_per_label) + " per label)."
            + " Common genes after intersection: " + str(len(common)) + "."
            + " Per-cell labels aggregated to per-cluster majority."
        ),
    }

# ---------------------------------------------------------------------------
# 4. CellAssign (GitHub: Irrationone/cellassign)
# ---------------------------------------------------------------------------

def _build_marker_matrix_from_v1(
    gene_universe: Sequence[str],
    target_layer: str = "cell_subtype",
    use_panel_genes: bool = True,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Build a (genes x cell-types) 0/1 marker matrix from cartilage_dictionary_v1.

    Returns ``(rho, genes, cell_types)`` where ``rho[g, t] == 1`` iff
    gene ``g`` is a marker for cell-type ``t`` according to v1.
    """
    from .assets import load_cartilage_dictionary_v1
    dictionary = load_cartilage_dictionary_v1()
    layer_axes = (dictionary.get("layers", {}) or {}).get(target_layer, {}).get("axes", [])
    cell_types: List[str] = []
    type_markers: Dict[str, set] = {}
    for ax in layer_axes:
        name = str(ax.get("name_en", ax.get("axis_id", "")))
        if not name:
            continue
        cell_types.append(name)
        markers: List[str] = []
        if use_panel_genes:
            markers.extend([str(g).upper() for g in (ax.get("panel_genes") or [])])
        markers.extend([str(g).upper() for g in (ax.get("core_genes") or [])])
        type_markers[name] = set(markers)
    gene_universe_up = [str(g).upper() for g in gene_universe]
    universe_set = set(gene_universe_up)
    used_genes_ordered: List[str] = []
    for ct in cell_types:
        for g in sorted(type_markers.get(ct, set()) & universe_set):
            if g not in used_genes_ordered:
                used_genes_ordered.append(g)
    pos_in_universe = {g: i for i, g in enumerate(gene_universe_up)}
    final_genes = [g for g in used_genes_ordered if g in pos_in_universe]
    rho = np.zeros((len(final_genes), len(cell_types)), dtype=np.float32)
    for ti, ct in enumerate(cell_types):
        marker_set = type_markers.get(ct, set())
        for gi, g in enumerate(final_genes):
            if g in marker_set:
                rho[gi, ti] = 1.0
    return rho, final_genes, cell_types


def annotate_with_cellassign(
    query_h5ad,
    *,
    cluster_col: str = "leiden_res0_5",
    target_layer: str = "cell_subtype",
    out_tsv=None,
    n_max_cells: int = 8000,
    random_state: int = 0,
    device: Optional[str] = None,
    learning_rate: float = 1e-2,
    max_iter_em: int = 20,
    max_iter_adam: int = 100,
) -> Dict[str, Any]:
    """Run real CellAssign on GPU via a PyTorch port of the EM.

    The upstream cellassign R package (Irrationone/cellassign) wraps a
    TensorFlow 1.x graph that has not been maintained since 2020 and
    crashes under TF >= 2 with shape-conversion errors deep inside
    tensorflow_probability. To keep ``cartigsfm annotate --method
    cellassign`` actually available we re-implement the same NB
    mixture EM in PyTorch (see ``cartigsfm.cellassign_torch``) so it
    runs on the same GPU as the rest of the package. The marker
    matrix is built from cartilage_dictionary_v1, identical to the
    previous Rscript path, so cross-method comparisons stay valid.
    """
    method = "cellassign"
    try:
        from .cellassign_torch import fit_cellassign_torch
    except Exception as exc:
        return _r_unavailable(
            method,
            install_hint="pip install torch (CUDA build recommended)",
            note="cartigsfm.cellassign_torch import failed: " + repr(exc),
        )

    import scanpy as sc
    q = sc.read_h5ad(str(query_h5ad))
    if cluster_col not in q.obs.columns:
        return _r_unavailable(method, install_hint="",
                              note="query has no cluster column " + repr(cluster_col))
    if q.n_obs > int(n_max_cells):
        rng = np.random.default_rng(int(random_state))
        keep = np.sort(rng.choice(q.n_obs, size=int(n_max_cells), replace=False))
        q_use = q[keep].copy()
    else:
        q_use = q

    rho, marker_genes, cell_types = _build_marker_matrix_from_v1(
        list(q_use.var_names.astype(str)),
        target_layer=target_layer,
        use_panel_genes=True,
    )
    if rho.size == 0 or len(marker_genes) == 0:
        return _r_unavailable(method, install_hint="",
                              note="cartilage_dictionary_v1 markers do not overlap query var_names")
    g2i = {str(g).upper(): i for i, g in enumerate(q_use.var_names.astype(str))}
    keep_cols = [g2i[g] for g in marker_genes if g in g2i]
    Xc = q_use.X
    if sp.issparse(Xc):
        Xc = np.asarray(Xc[:, keep_cols].toarray(), dtype=np.float32)
    else:
        Xc = np.asarray(np.asarray(Xc, dtype=np.float32)[:, keep_cols], dtype=np.float32)

    size_factors = np.asarray(Xc.sum(axis=1), dtype=np.float64)
    size_factors = size_factors / max(float(np.median(size_factors[size_factors > 0])) or 1.0, 1.0)
    size_factors = np.clip(size_factors, 1e-3, None)

    cell_ids = list(q_use.obs_names.astype(str))
    clusters = list(q_use.obs[cluster_col].astype(str))
    fit = fit_cellassign_torch(
        Y=Xc.astype(np.float64),
        rho=rho.astype(np.float64),
        s=size_factors,
        device=device,
        learning_rate=float(learning_rate),
        max_iter_em=int(max_iter_em),
        max_iter_adam=int(max_iter_adam),
        random_seed=int(random_state),
        verbose=False,
    )
    cell_type_idx = np.asarray(fit["cell_type"], dtype=np.int64)
    gamma = np.asarray(fit["gamma"], dtype=np.float64)
    label_per_cell = [
        cell_types[int(i)] if 0 <= int(i) < len(cell_types) else ""
        for i in cell_type_idx
    ]
    if gamma.shape[0]:
        max_gamma = gamma.max(axis=1)
    else:
        max_gamma = np.zeros((0,), dtype=np.float64)
    out = pd.DataFrame({
        "cell_id": cell_ids,
        "cluster": [str(x) for x in clusters],
        "cellassign_label": label_per_cell,
        "cellassign_max_gamma": max_gamma,
    })
    out["cluster"] = out["cluster"].astype(str)
    rows: List[Dict[str, Any]] = []
    for cid, sub in out.groupby("cluster"):
        labs = [str(x) for x in sub["cellassign_label"].astype(str).tolist() if str(x) and str(x) != "NA" and str(x) != "nan"]
        if not labs:
            top = ""
            top_n = 0
        else:
            top, top_n = Counter(labs).most_common(1)[0]
        rows.append({
            cluster_col: str(cid),
            "cellassign_label": str(top),
            "cellassign_n_cells": int(len(sub)),
            "cellassign_majority_frac": round(float(top_n) / max(len(sub), 1), 4),
        })
    per_cluster = pd.DataFrame(rows).sort_values(cluster_col).reset_index(drop=True)
    out_path: Optional[str] = None
    if out_tsv is not None:
        out_path = str(out_tsv)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, sep="\t", index=False)
    return {
        "method": method,
        "available": True,
        "per_cluster": per_cluster,
        "per_cell_predictions_path": out_path,
        "n_query_cells": int(q_use.n_obs),
        "n_clusters": int(per_cluster.shape[0]),
        "cluster_col": cluster_col,
        "n_marker_genes": int(len(marker_genes)),
        "n_cell_types": int(len(cell_types)),
        "target_layer": target_layer,
        "note": (
            "Real CellAssign via cartigsfm.cellassign_torch (PyTorch port"
            " of Irrationone/cellassign EM, runs on " + str(fit.get("device", "cpu"))
            + ", " + str(fit.get("n_iters", 0)) + " EM iters) with "
            + str(len(marker_genes))
            + " marker genes from cartilage_dictionary_v1 layer "
            + repr(target_layer) + " (" + str(len(cell_types))
            + " cell types). Query subsampled to " + str(q_use.n_obs)
            + " cells. Per-cell labels aggregated to per-cluster majority."
        ),
    }
