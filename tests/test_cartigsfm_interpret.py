"""Unit tests for cartigsfm.interpret.

Run with:  python -m unittest tests.test_cartigsfm_interpret -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cartigsfm import interpret as I


class TestAxisSafetyClass(unittest.TestCase):
    def test_status_to_safety_mapping(self):
        self.assertEqual(I.axis_safety_class({"status": "production"}), "PENDING_INDEPENDENT_VALIDATION")
        self.assertEqual(I.axis_safety_class({"status": "reference"}), "SUPPLEMENTARY_ONLY")
        self.assertEqual(I.axis_safety_class({"status": "literature_prior"}), "EXPLORATORY")
        self.assertEqual(I.axis_safety_class({"status": "weird_status"}), "PENDING_INDEPENDENT_VALIDATION")
        self.assertEqual(
            I.axis_safety_class({"status": "reference", "safety_classification": "MAIN_TEXT_READY"}),
            "MAIN_TEXT_READY")


class TestClassifyClaim(unittest.TestCase):
    def test_exact_match_blocks_LLM_claim(self):
        result = I.classify_claim("CartiGSFM is a trained cartilage large language model (LLM)")
        self.assertEqual(result["safety_classification"], "NOT_SUPPORTED")
        self.assertFalse(result["can_claim"])

    def test_exact_match_blocks_OA_inflammation_overclaim(self):
        result = I.classify_claim("Inflammation_NFkB, IL1_Signaling, and TNF_Signaling are significantly increased in OA")
        self.assertEqual(result["safety_classification"], "NOT_SUPPORTED")
        self.assertFalse(result["can_claim"])

    def test_regex_guard_blocks_external_validation(self):
        result = I.classify_claim("This axis is externally validated in our cohort")
        self.assertEqual(result["safety_classification"], "NOT_SUPPORTED")
        self.assertFalse(result["can_claim"])
        self.assertIn("matched_pattern", result)

    def test_regex_guard_blocks_therapeutic_target(self):
        result = I.classify_claim("MGP is a therapeutic target for cartilage disease")
        self.assertEqual(result["safety_classification"], "NOT_SUPPORTED")
        self.assertFalse(result["can_claim"])

    def test_regex_guard_blocks_LLM_keyword(self):
        result = I.classify_claim("We trained CartiGSFM as a cartilage LLM")
        self.assertEqual(result["safety_classification"], "NOT_SUPPORTED")
        self.assertFalse(result["can_claim"])

    def test_conservative_claim_passes_unreviewed(self):
        result = I.classify_claim("Avascular_Antimineralization score is associated with sample S1")
        self.assertEqual(result["safety_classification"], "UNREVIEWED")
        self.assertTrue(result["can_claim"])

    def test_empty_claim_blocked(self):
        result = I.classify_claim("")
        self.assertFalse(result["can_claim"])
        self.assertEqual(result["safety_classification"], "UNREVIEWED")



class TestInterpretGeneList(unittest.TestCase):
    def test_avam_canonical_genes_top_hit(self):
        genes = ["MGP", "CNMD", "LECT1", "TIMP3", "ANKH", "ENPP1",
                 "TNFRSF11B", "FRZB", "SOX9", "ACAN"]
        result = I.interpret_gene_list(genes, top_per_layer=3)
        self.assertEqual(result["mode"], "genes")
        self.assertGreater(result["axis_count_scored"], 5)
        self.assertGreater(result["axis_count_kept"], 0)
        self.assertIn("safety_summary", result)
        top = result["overall_top_axes"]
        self.assertGreater(len(top), 0)
        self.assertEqual(top[0]["axis_id"], "functional_axis::Avascular_Antimineralization")
        self.assertEqual(top[0]["safety_classification"], "SUPPLEMENTARY_ONLY")
        self.assertGreater(top[0]["score"], 0.0)
        self.assertEqual(result["warnings"], [])

    def test_empty_gene_list_returns_warning(self):
        result = I.interpret_gene_list([])
        self.assertEqual(result["axis_count_scored"], 0)
        self.assertEqual(result["axis_count_kept"], 0)
        self.assertIn("Empty gene list after normalization.", result["warnings"])

    def test_no_overlap_returns_warning(self):
        result = I.interpret_gene_list(["ZZZ_UNUSUAL_GENE_1", "ZZZ_UNUSUAL_GENE_2"])
        self.assertEqual(result["axis_count_scored"], 0)
        self.assertIn("warnings", result)
        self.assertGreater(len(result["warnings"]), 0)


class TestInterpretP4Csv(unittest.TestCase):
    def test_p4_csv_with_three_layers(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            df = pd.DataFrame([
                {"axis_id": "functional_axis::Avascular_Antimineralization",
                 "layer": "functional_axis", "sample": "S1|ear|C0", "score": 0.85},
                {"axis_id": "functional_axis::Inflammation_NFkB",
                 "layer": "functional_axis", "sample": "S1|ear|C0", "score": 0.12},
                {"axis_id": "tissue_developmental_state::ElasticCartilage_Auricular",
                 "layer": "tissue_developmental_state", "sample": "S1|ear|C0", "score": 1.20},
                {"axis_id": "cell_subtype::Homeostatic_Chondrocytes",
                 "layer": "cell_subtype", "sample": "S1|ear|C0", "score": 0.45},
            ])
            csv_path = root / "scores.tsv"
            df.to_csv(csv_path, sep="\t", index=False)
            result = I.interpret_p4_csv(csv_path, top_per_layer=2)
            self.assertEqual(result["mode"], "p4_csv")
            self.assertEqual(result["input"]["n_rows"], 4)
            self.assertEqual(result["axis_count_scored"], 4)
            self.assertEqual(result["axis_count_kept"], 4)
            self.assertEqual(set(result["safety_summary"].keys()),
                             {"PENDING_INDEPENDENT_VALIDATION", "SUPPLEMENTARY_ONLY", "EXPLORATORY"})
            df_bad = pd.DataFrame([{"axis_id": "fake::NotARealAxis", "layer": "fake",
                                     "sample": "S1", "score": 1.0}])
            bad_path = root / "bad.tsv"
            df_bad.to_csv(bad_path, sep="\t", index=False)
            result_bad = I.interpret_p4_csv(bad_path)
            self.assertEqual(result_bad["axis_count_scored"], 0)
            self.assertTrue(any("not in cartilage_dictionary_v1" in w
                                for w in result_bad["warnings"]))

    def test_p4_csv_missing_columns_raises(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            df = pd.DataFrame({"foo": [1, 2, 3]})
            bad_path = root / "bad.tsv"
            df.to_csv(bad_path, sep="\t", index=False)
            with self.assertRaises(KeyError):
                I.interpret_p4_csv(bad_path)


class TestConfidenceAndExperiment(unittest.TestCase):
    def test_confidence_high_when_avam_overlap(self):
        genes = ["MGP", "CNMD", "LECT1", "TIMP3", "ANKH", "ENPP1",
                 "TNFRSF11B", "FRZB", "SOX9", "ACAN"]
        result = I.interpret_gene_list(genes, top_per_layer=2)
        av = next(a for a in result["overall_top_axes"]
                  if a["axis_id"].endswith("Avascular_Antimineralization"))
        self.assertEqual(av["confidence"]["label"], "high")
        self.assertGreater(av["confidence"]["value"], 0.30)
        self.assertIn("core_genes are present", av["confidence"]["basis"])

    def test_confidence_low_when_no_overlap(self):
        result = I.interpret_gene_list(["ZZZ_FAKE1", "ZZZ_FAKE2"])
        self.assertEqual(result["overall_top_axes"], [])

    def test_confidence_p4_path_uses_n_samples(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            df = pd.DataFrame([
                {"axis_id": "functional_axis::Avascular_Antimineralization",
                 "layer": "functional_axis", "sample": f"S{i}|ear|C0",
                 "score": 0.6 + 0.01 * i} for i in range(5)
            ])
            p2 = root / "p4.tsv"
            df.to_csv(p2, sep="\t", index=False)
            r = I.interpret_p4_csv(p2, top_per_layer=1)
            av = next(a for a in r["overall_top_axes"]
                      if a["axis_id"].endswith("Avascular_Antimineralization"))
            self.assertEqual(av["confidence"]["label"], "high")
            self.assertIn("n_samples=5", av["confidence"]["basis"])

    def test_suggested_experiment_layer_specific(self):
        d = I.load_cartilage_dictionary_v1()
        cell_subtype = next(a for _, a in I._iter_v1_axes(d)
                            if a.get("layer") == "cell_subtype")
        tissue = next(a for _, a in I._iter_v1_axes(d)
                      if a.get("layer") == "tissue_developmental_state")
        func = next(a for _, a in I._iter_v1_axes(d)
                    if a.get("layer") == "functional_axis")
        self.assertIn("Flow cytometry", I.suggested_validation_experiment(cell_subtype))
        self.assertIn("Histology", I.suggested_validation_experiment(tissue))
        self.assertIn("Pathway-level", I.suggested_validation_experiment(func))

    def test_render_markdown_includes_confidence_and_experiment(self):
        genes = ["MGP", "CNMD", "TIMP3", "ANKH", "ENPP1", "TNFRSF11B"]
        result = I.interpret_gene_list(genes, top_per_layer=2)
        md = I.render_markdown(result)
        self.assertIn("confidence:", md)
        self.assertIn("suggested validation experiment:", md)
        self.assertIn("Pathway-level qPCR", md)

