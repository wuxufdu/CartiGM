"""Unit tests for the streaming pseudobulk path used for atlas-scale h5ads."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cartigsfm.p4 import (
    _autostream_threshold_bytes,
    _resolve_chunk_size,
    pseudobulk_from_h5ad,
    pseudobulk_streaming,
)


def _make_synthetic_h5ad(path: Path, *, n_cells: int, n_genes: int,
                          sample_per_cell, tissue_per_cell, cluster_per_cell,
                          density: float = 0.05, seed: int = 0):
    """Write a small sparse h5ad with a hand-built design we can verify against."""
    import anndata as ad
    from scipy import sparse
    rng = np.random.default_rng(seed)
    n_nonzero = max(int(n_cells * n_genes * density), n_cells)
    rows = rng.integers(0, n_cells, size=n_nonzero)
    cols = rng.integers(0, n_genes, size=n_nonzero)
    vals = rng.poisson(2.0, size=n_nonzero).astype(np.float32)
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_genes))
    obs = pd.DataFrame({
        "sample": sample_per_cell,
        "tissue": tissue_per_cell,
        "cluster": cluster_per_cell,
        "celltype": ["Chondrocyte"] * n_cells,
    })
    var = pd.DataFrame(index=[f"GENE{i:03d}" for i in range(n_genes)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.write_h5ad(path)


class TestChunkSizeResolver(unittest.TestCase):
    def test_resolve_chunk_size_clamps_to_floor_and_ceiling(self):
        # 1M genes would need a tiny chunk; clamped to floor
        self.assertEqual(_resolve_chunk_size(1_000_000), 1000)
        # 100 genes fits the ceiling
        self.assertEqual(_resolve_chunk_size(100), 20000)

    def test_autostream_threshold_is_2gb(self):
        self.assertEqual(_autostream_threshold_bytes(), 2 * 1024 ** 3)


class TestStreamingMatchesInMemory(unittest.TestCase):
    """The streaming and in-memory paths must produce numerically close means."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.h5ad = Path(self.tmp.name) / "toy.h5ad"
        n_cells, n_genes = 600, 200
        samples = np.array([f"S{i // 200}" for i in range(n_cells)])
        tissues = np.array(["ear" if i % 2 == 0 else "rib" for i in range(n_cells)])
        clusters = np.array([f"C{i // 100}" for i in range(n_cells)])
        _make_synthetic_h5ad(
            self.h5ad, n_cells=n_cells, n_genes=n_genes,
            sample_per_cell=samples, tissue_per_cell=tissues,
            cluster_per_cell=clusters, density=0.1, seed=42,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_streaming_returns_same_groups_as_in_memory(self):
        pb_mem, meta_mem = pseudobulk_from_h5ad(
            self.h5ad, sample_col="sample", tissue_col="tissue",
            cluster_col="cluster", celltype_col="celltype",
            celltype_regex="chondro",
        )
        pb_str, meta_str = pseudobulk_streaming(
            self.h5ad, sample_col="sample", tissue_col="tissue",
            cluster_col="cluster", celltype_col="celltype",
            celltype_regex="chondro", chunk_size=128,
        )
        self.assertEqual(set(pb_mem.columns), set(pb_str.columns))
        self.assertEqual(set(meta_mem.index), set(meta_str.index))
        # every cell accounted for
        self.assertEqual(int(meta_mem["n_cells"].sum()),
                         int(meta_str["n_cells"].sum()))
        # means should match within float32 tolerance
        common = sorted(c for c in (set(pb_mem.columns) & set(pb_str.columns))
                        if c != "gene")
        a = pb_mem.set_index("gene")[common].to_numpy()
        b = pb_str.set_index("gene")[common].to_numpy()
        np.testing.assert_allclose(a, b, atol=1e-4, rtol=1e-4)

    def test_streaming_respects_min_cells_filter(self):
        with self.assertRaises(ValueError) as cm:
            pseudobulk_streaming(
                self.h5ad, sample_col="sample", tissue_col="tissue",
                cluster_col="cluster", celltype_col="celltype",
                celltype_regex="chondro", chunk_size=128, min_cells=10_000,
            )
        self.assertIn("no sample-cluster groups", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
