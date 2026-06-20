"""Extract a 80k-cell HVG-restricted training subset from acc_new.h5ad and a
full HVG-restricted dump of EBR.h5ad on the remote, then SFTP small npz
files back to the local 4090 host.
"""
from __future__ import annotations

from pathlib import Path
import paramiko

HOST = "sw2-dynamic.xiyoucloud.pro"; PORT = 12264
USER = "wuxu"; PASSWORD = "Wuxu@96885"
REMOTE_PY = "/home/wuxu/miniconda3/envs/squidp311/bin/python"
REMOTE_SCRIPT = "/tmp/_extract_train_subset.py"
REMOTE_OUT_ACC = "/tmp/acc_train.npz"
REMOTE_OUT_EBR = "/tmp/ebr_eval.npz"
REMOTE_OUT_HVG = "/tmp/hvg_genes.tsv"
REMOTE_LOG = "/tmp/extract_train_subset.log"
ACC_H5 = "/home/wuxu/jupyter/MJ/newh5ad/acc_new.h5ad"
EBR_H5 = "/home/wuxu/jupyter/MJ/newh5ad/EBL/EBR.h5ad"

LOCAL_OUT = Path("F:/cartifm/outputs/training_subset")

REMOTE_CODE = """
import os, sys, time
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
import numpy as np
import scanpy as sc
import anndata as ad
import scipy.sparse as sp

ACC = '__ACC_H5__'
EBR = '__EBR_H5__'
OUT_ACC = '__OUT_ACC__'
OUT_EBR = '__OUT_EBR__'
OUT_HVG = '__OUT_HVG__'
N_CELLS = 80000
N_HVG = 2000
SEED = 0
LABEL_COL = 'celltype_new'
SAMPLE_COL = 'sample'

rng = np.random.default_rng(SEED)

# First, peek EBR.var_names so we restrict acc HVG selection to the
# acc ∩ EBR gene intersection (no zero-padded columns at train time).
print('peeking EBR var_names ...', flush=True)
e_var = ad.read_h5ad(EBR, backed='r').var_names.tolist()
ebr_var_set = set(e_var)
print('EBR var size', len(ebr_var_set), flush=True)

print('reading acc_new ...', flush=True)
a = ad.read_h5ad(ACC, backed='r')
labels = a.obs[LABEL_COL].astype(str).values
samples = a.obs[SAMPLE_COL].astype(str).values
print('acc shape', a.shape, 'labels', np.unique(labels).size, flush=True)
# acc-only var_names mask
acc_in_ebr = np.array([g in ebr_var_set for g in a.var_names])
print('acc ∩ EBR genes =', int(acc_in_ebr.sum()), '/', len(a.var_names), flush=True)
groups = np.array([f"{s}|{l}" for s, l in zip(samples, labels)])
uniq = np.unique(groups)
per = max(1, N_CELLS // len(uniq))
keep = []
for g in uniq:
    idx = np.where(groups == g)[0]
    k = min(per, len(idx))
    if k >= len(idx):
        keep.append(idx)
    else:
        keep.append(rng.choice(idx, size=k, replace=False))
keep = np.sort(np.concatenate(keep))
if keep.size > N_CELLS:
    keep = np.sort(rng.choice(keep, size=N_CELLS, replace=False))
print('subsampled keep', keep.size, flush=True)
a_sub = a[keep].to_memory()
print('to_memory ok', a_sub.shape, flush=True)

# acc_new.X is already log1p-normalized (probed max ~7.83). Use it as-is to
# stay on the same scale as EBR.layers["log1p_norm"].
sc.pp.highly_variable_genes(a_sub, n_top_genes=N_HVG, flavor='seurat')
# Force HVG to live inside the acc ∩ EBR intersection by re-flagging.
raw_hvg = np.where(a_sub.var['highly_variable'].values)[0]
# also need a fallback ranking by dispersion in case fewer than N_HVG remain
disp_col = None
for cand in ('dispersions_norm', 'dispersions'):
    if cand in a_sub.var.columns:
        disp_col = cand; break
disp_vals = a_sub.var[disp_col].values if disp_col else np.zeros(a_sub.n_vars)
rank_order = np.argsort(-disp_vals)
# filter to acc ∩ EBR intersection
ranked_filtered = [i for i in rank_order if acc_in_ebr[i]]
keep_top = ranked_filtered[:N_HVG]
a_sub.var['highly_variable'] = False
a_sub.var.iloc[keep_top, a_sub.var.columns.get_loc('highly_variable')] = True
print('HVG (intersection-filtered)', int(a_sub.var['highly_variable'].sum()), flush=True)
hvg_mask = a_sub.var['highly_variable'].values
hvg_idx = np.where(hvg_mask)[0]
if hvg_idx.size > N_HVG:
    hvg_idx = hvg_idx[:N_HVG]
gene_names = np.asarray(a_sub.var_names)[hvg_idx]
print('hvg', hvg_idx.size, flush=True)

X_acc = a_sub.X[:, hvg_idx]
X_acc = X_acc.toarray().astype(np.float32) if sp.issparse(X_acc) else np.asarray(X_acc, dtype=np.float32)
print('X_acc', X_acc.shape, 'mem MB', round(X_acc.nbytes/1e6, 1), flush=True)

np.savez_compressed(OUT_ACC,
                    X=X_acc,
                    labels=labels[keep].astype(str),
                    samples=samples[keep].astype(str),
                    genes=gene_names.astype(str))
print('wrote', OUT_ACC, flush=True)

with open(OUT_HVG, 'w') as f:
    f.write('gene\\tindex\\n')
    for i, g in enumerate(gene_names):
        f.write(f"{g}\\t{i}\\n")

print('reading EBR ...', flush=True)
e = ad.read_h5ad(EBR)
print('EBR shape', e.shape, flush=True)
# EBR X is scVelo-polluted (negative values). Use layers['log1p_norm'] which
# is the same scale as acc_new.X.
if 'log1p_norm' in e.layers:
    e.X = e.layers['log1p_norm']
    print('EBR layer log1p_norm -> X', flush=True)

ebr_var = np.asarray(e.var_names)
name_to_idx = {g: i for i, g in enumerate(ebr_var)}
col_map = []
ok_cols = []
for i, g in enumerate(gene_names):
    j = name_to_idx.get(g)
    if j is not None:
        col_map.append(j)
        ok_cols.append(i)
col_map = np.asarray(col_map, dtype=np.int64)
ok_cols = np.asarray(ok_cols, dtype=np.int64)
print('EBR shared HVGs', col_map.size, '/', gene_names.size, flush=True)

X_ebr = e.X[:, col_map]
X_ebr = X_ebr.toarray().astype(np.float32) if sp.issparse(X_ebr) else np.asarray(X_ebr, dtype=np.float32)
if ok_cols.size < gene_names.size:
    Xfull = np.zeros((X_ebr.shape[0], gene_names.size), dtype=np.float32)
    Xfull[:, ok_cols] = X_ebr
    X_ebr = Xfull
print('X_ebr', X_ebr.shape, 'mem MB', round(X_ebr.nbytes/1e6, 1), flush=True)

np.savez_compressed(OUT_EBR,
                    X=X_ebr,
                    celltype=e.obs['celltype'].astype(str).values,
                    batch=e.obs['batch'].astype(str).values,
                    cluster=e.obs['leiden_res1'].astype(str).values,
                    genes=gene_names.astype(str))
print('wrote', OUT_EBR, flush=True)
print('done', flush=True)
"""


def main() -> None:
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)

    code = (REMOTE_CODE
            .replace("__ACC_H5__", ACC_H5)
            .replace("__EBR_H5__", EBR_H5)
            .replace("__OUT_ACC__", REMOTE_OUT_ACC)
            .replace("__OUT_EBR__", REMOTE_OUT_EBR)
            .replace("__OUT_HVG__", REMOTE_OUT_HVG))

    print("Connecting", HOST)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD,
              timeout=30, look_for_keys=False, allow_agent=False)

    sftp = c.open_sftp()
    with sftp.open(REMOTE_SCRIPT, "w") as f:
        f.write(code)
    sftp.close()

    cmd = (
        f"HDF5_USE_FILE_LOCKING=FALSE {REMOTE_PY} {REMOTE_SCRIPT} "
        f"> {REMOTE_LOG} 2>&1; echo DONE_RC=$?"
    )
    print("$", cmd)
    _, o, e = c.exec_command(cmd, timeout=7200)
    print(o.read().decode(errors='replace'))
    err = e.read().decode(errors='replace').rstrip()
    if err:
        print("[stderr]", err)

    print("\n=== tail log ===")
    _, o, _ = c.exec_command(f"tail -60 {REMOTE_LOG}", timeout=30)
    print(o.read().decode(errors='replace'))
    _, o, _ = c.exec_command(f"ls -lh {REMOTE_OUT_ACC} {REMOTE_OUT_EBR} {REMOTE_OUT_HVG}", timeout=30)
    print(o.read().decode(errors='replace'))

    sftp = c.open_sftp()
    for remote, local in [
        (REMOTE_OUT_ACC, LOCAL_OUT / "acc_train.npz"),
        (REMOTE_OUT_EBR, LOCAL_OUT / "ebr_eval.npz"),
        (REMOTE_OUT_HVG, LOCAL_OUT / "hvg_genes.tsv"),
        (REMOTE_LOG, LOCAL_OUT / "extract.log"),
    ]:
        try:
            sftp.get(remote, str(local))
            print("pulled", remote, "->", local)
        except Exception as exc:
            print("skip", remote, exc)
    sftp.close()
    c.close()
    print("ok ->", LOCAL_OUT)


if __name__ == "__main__":
    main()
