"""Unit tests for cartigsfm.gsfm (P12: GSFM-style frozen gene-set branch).

Run with: python -m unittest tests.test_gsfm -v
"""
from __future__ import annotations

import unittest

from cartigsfm import gsfm


AVAM_AXIS = "functional_axis::Avascular_Antimineralization"
AVAM_MARKERS = [
    "MGP", "CNMD", "TIMP3", "ANKH", "ENPP1",
    "TNFRSF11B", "FRZB", "SOX9", "ACAN",
]


class TestGsfmAxisTable(unittest.TestCase):
    def test_axis_table_shape(self):
        t = gsfm.gsfm_axis_table()
        # Floor reflects the v1.0 release; the dictionary may add axes
        # (e.g. Nasal_Septum_Cartilage in v1.2) without breaking shape contract.
        self.assertGreaterEqual(t.shape[0], 42)
        for col in ("axis_id", "layer", "n_core", "n_anti",
                    "has_literature_support", "safety_classification"):
            self.assertIn(col, t.columns)

    def test_avam_axis_present_with_expected_features(self):
        t = gsfm.gsfm_axis_table()
        row = t[t["axis_id"] == AVAM_AXIS].iloc[0]
        self.assertGreater(int(row["n_core"]), 5)
        self.assertEqual(row["layer"], "functional_axis")


class TestGsfmAxisEmbedding(unittest.TestCase):
    def test_avam_embedding_contains_core_genes(self):
        e = gsfm.gsfm_axis_embedding(AVAM_AXIS)
        self.assertTrue(e["found"])
        for marker in ("MGP", "TIMP3", "ANKH"):
            self.assertIn(marker, e["core_genes"])

    def test_missing_axis_returns_safe_default(self):
        e = gsfm.gsfm_axis_embedding("functional_axis::DOES_NOT_EXIST")
        self.assertFalse(e["found"])
        self.assertEqual(e["safety_classification"], "UNREVIEWED")


class TestGsfmSimilarity(unittest.TestCase):
    def test_avam_similarity_high_for_canonical_markers(self):
        s = gsfm.gsfm_axis_similarity(AVAM_MARKERS, AVAM_AXIS)
        self.assertGreater(s, 0.3)
        self.assertLessEqual(s, 1.0)

    def test_similarity_zero_for_empty_query(self):
        self.assertEqual(gsfm.gsfm_axis_similarity([], AVAM_AXIS), 0.0)
        self.assertEqual(gsfm.gsfm_axis_similarity(None, AVAM_AXIS), 0.0)

    def test_similarity_zero_for_unknown_axis(self):
        s = gsfm.gsfm_axis_similarity(AVAM_MARKERS, "functional_axis::NOPE")
        self.assertEqual(s, 0.0)

    def test_marker_axes_top_ranks_avam_for_calcification_inhibitors(self):
        top = gsfm.gsfm_marker_axes(AVAM_MARKERS, top_n=3)
        self.assertGreater(len(top), 0)
        self.assertEqual(top[0]["axis_id"], AVAM_AXIS)
        self.assertGreater(top[0]["shared_n"], 3)


class TestGsfmToolWrapper(unittest.TestCase):
    def test_tool_gsfm_score_returns_branch_dict(self):
        r = gsfm.tool_gsfm_score(AVAM_MARKERS)
        self.assertEqual(r["branch"], "gsfm")
        self.assertIn("result", r)
        self.assertIn("top_axes", r["result"])
        self.assertGreater(len(r["result"]["top_axes"]), 0)

    def test_tool_gsfm_score_with_axis_id(self):
        r = gsfm.tool_gsfm_score(AVAM_MARKERS, axis_id=AVAM_AXIS)
        self.assertEqual(r["axis_id"], AVAM_AXIS)
        self.assertIn("similarity", r["result"])
        self.assertIn("embedding", r["result"])
        self.assertGreater(r["result"]["similarity"], 0.0)

    def test_tool_gsfm_score_empty_input(self):
        r = gsfm.tool_gsfm_score([])
        self.assertIsNone(r["result"])
        self.assertIn("empty", r["note"])
