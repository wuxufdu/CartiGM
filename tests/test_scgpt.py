"""Unit tests for cartigsfm.scgpt (P13: scGPT-style frozen expression branch).

Run with: python -m unittest tests.test_scgpt -v
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cartigsfm import scgpt


def _toy_expression() -> pd.DataFrame:
    """A 14-gene, 2-cluster toy expression matrix that exercises core_genes."""
    return pd.DataFrame({
        "gene": [
            "MGP", "CNMD", "TIMP3", "ANKH", "ENPP1",
            "TNFRSF11B", "FRZB", "SOX9", "ACAN", "COL2A1",
            "ELN", "RUNX2", "COL10A1", "MMP13",
        ],
        "C0_ear": [10, 9, 8, 7, 7, 6, 6, 5, 5, 4, 3, 0.5, 0.5, 0.5],
        "C1_rib": [1, 1, 1, 1, 1, 0.5, 0.5, 5, 5, 6, 0.5, 7, 7, 7],
    })


class TestScgptDataFrame(unittest.TestCase):
    def test_encode_dataframe_returns_required_keys(self):
        r = scgpt.scgpt_encode_dataframe(_toy_expression(), gene_col="gene")
        self.assertEqual(r["branch"], "scgpt")
        self.assertEqual(r["input_kind"], "dataframe")
        self.assertEqual(r["n_clusters"], 2)
        self.assertGreater(r["n_axes_scored"], 0)
        self.assertIn("axis_scores", r)
        self.assertIn("cluster_embedding", r)

    def test_cluster_embedding_has_one_row_per_cluster(self):
        r = scgpt.scgpt_encode_dataframe(_toy_expression(), gene_col="gene")
        emb = r["cluster_embedding"]
        self.assertEqual(emb.shape[0], 2)
        self.assertIn("top_axis_id", emb.columns)
        self.assertIn("top_score", emb.columns)

    def test_each_cluster_has_top_axis_with_score(self):
        r = scgpt.scgpt_encode_dataframe(_toy_expression(), gene_col="gene")
        emb = r["cluster_embedding"]
        for cluster in emb.index:
            self.assertTrue(str(emb.loc[cluster, "top_axis_id"]))
            self.assertGreater(float(emb.loc[cluster, "top_score"]), 0.0)

    def test_missing_columns_raises(self):
        bad = pd.DataFrame({"gene": ["MGP"], "X": [1.0]})
        with self.assertRaises(ValueError):
            scgpt.scgpt_encode_dataframe(bad, sample_cols=["Y"])

    def test_empty_expression_returns_zero_axes(self):
        empty = pd.DataFrame({"gene": ["FOO", "BAR"], "C0": [0.0, 0.0]})
        r = scgpt.scgpt_encode_dataframe(empty, gene_col="gene")
        self.assertEqual(r["n_axes_scored"], 0)


class TestScgptToolWrapper(unittest.TestCase):
    def test_tool_scgpt_encode_dataframe(self):
        r = scgpt.tool_scgpt_encode(expr_df=_toy_expression(), gene_col="gene")
        self.assertEqual(r["branch"], "scgpt")
        self.assertEqual(len(r["per_cluster_summary"]), 2)
        for entry in r["per_cluster_summary"]:
            self.assertIn("cluster", entry)
            self.assertIn("top_axis_id", entry)
            self.assertGreater(len(entry["top_axes"]), 0)

    def test_tool_scgpt_encode_no_input(self):
        r = scgpt.tool_scgpt_encode()
        self.assertIsNone(r["result"])
        self.assertIn("provide either", r["note"])
