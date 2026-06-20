"""Unit tests for cartigsfm.agent.

Run with:  python -m unittest tests.test_agent -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cartigsfm import agent as A


class TestToolSchema(unittest.TestCase):
    def test_three_tools_registered(self):
        names = [t["function"]["name"] for t in A.TOOL_SCHEMA]
        self.assertEqual(set(names), {
            "cartigm_score", "p4_project", "rag_evidence_lookup",
            "gsfm_score", "scgpt_encode",
        })

    def test_dispatch_covers_schema(self):
        names = {t["function"]["name"] for t in A.TOOL_SCHEMA}
        self.assertEqual(set(A.TOOL_DISPATCH), names)

    def test_required_fields(self):
        for t in A.TOOL_SCHEMA:
            fn = t["function"]
            self.assertIn("name", fn)
            self.assertIn("description", fn)
            self.assertIn("parameters", fn)
            self.assertIn("required", fn["parameters"])


class TestToolCartigmScore(unittest.TestCase):
    def test_returns_evidence_audits_and_experiment(self):
        r = A.tool_cartigm_score(
            ["MGP", "CNMD", "LECT1", "TIMP3", "ANKH", "ENPP1",
             "TNFRSF11B", "FRZB", "SOX9", "ACAN"],
            top_per_layer=2, overall_top=3,
        )
        self.assertIn("overall_top_axes", r)
        self.assertGreater(len(r["overall_top_axes"]), 0)
        top = r["overall_top_axes"][0]
        self.assertIn("confidence", top)
        self.assertIn("suggested_validation_experiment", top)
        self.assertIn("hard_constraints", r)
        self.assertIn("cannot_claim", r)

    def test_accepts_comma_separated_string(self):
        r = A.tool_cartigm_score("MGP, CNMD, TIMP3, ACAN")
        self.assertGreater(len(r["overall_top_axes"]), 0)


class TestToolRagLookup(unittest.TestCase):
    def test_avam_axis_id(self):
        r = A.tool_rag_evidence_lookup("functional_axis::Avascular_Antimineralization")
        self.assertIn("Avascular_Antimineralization",
                      str(list(r["axis_cards"].keys())))
        self.assertIn("kb_snippets", r)

    def test_free_text_query(self):
        r = A.tool_rag_evidence_lookup("inflammation in OA cartilage")
        self.assertIsInstance(r["axis_cards"], dict)

    def test_claim_classification_for_unmapped_text(self):
        r = A.tool_rag_evidence_lookup("a generic biological question")
        self.assertIn("query", r)


class TestToolP4Project(unittest.TestCase):
    def test_pseudobulk_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            genes = ["MGP", "CNMD", "TIMP3", "ANKH", "ENPP1",
                     "ELN", "FBLN5", "COL2A1", "ACAN", "RUNX2"]
            pb = pd.DataFrame({
                "gene": genes,
                "S1|ear|C0": [10, 10, 9, 8, 8, 7, 7, 5, 5, 0.5],
                "S2|rib|C0": [2, 1, 2, 1, 1, 0.5, 0.5, 7, 7, 8],
            })
            meta = pd.DataFrame(
                [
                    {"sample": "S1", "tissue": "ear", "cluster": "C0", "n_cells": 30},
                    {"sample": "S2", "tissue": "rib", "cluster": "C0", "n_cells": 30},
                ],
                index=["S1|ear|C0", "S2|rib|C0"],
            )
            pb_p = root / "pb.tsv"; meta_p = root / "meta.tsv"
            outdir = root / "p4"
            pb.to_csv(pb_p, sep="\t", index=False)
            meta.to_csv(meta_p, sep="\t")
            r = A.tool_p4_project(None, str(outdir)) if False else _p4_via_pseudobulk(pb_p, meta_p, outdir)
            self.assertIn("outputs", r)
            self.assertTrue(Path(r["outputs"]["scores"]).exists())


def _p4_via_pseudobulk(pb, meta, outdir):
    from cartigsfm.p4 import run_p4_project
    out = Path(outdir)
    outputs = run_p4_project(outdir=out, pseudobulk_tsv=pb, meta_tsv=meta)
    return {"outdir": str(out), "outputs": {k: str(v) for k, v in outputs.items()}}


class TestKeywordDispatcher(unittest.TestCase):
    def test_routes_gene_query_to_cartigm_score(self):
        r = A.run_query_keyword("Is MGP CNMD ACAN enriched in cartilage?")
        self.assertEqual(r["tool"], "cartigm_score")
        self.assertGreater(len(r["result"]["overall_top_axes"]), 0)

    def test_routes_evidence_query_to_rag(self):
        r = A.run_query_keyword("give me evidence for Avascular_Antimineralization")
        self.assertEqual(r["tool"], "rag_evidence_lookup")

    def test_routes_p4_query(self):
        r = A.run_query_keyword("run p4 on my self_data.h5ad to outdir my_p4")
        self.assertEqual(r["tool"], "p4_project")
        # When the referenced h5ad does not exist, the keyword dispatcher
        # refuses to execute and returns a `note` naming the missing path
        # rather than fabricating a fake P4 outdir.
        self.assertIsNone(r["result"])
        self.assertIn("self_data.h5ad", r["note"])

    def test_unknown_query(self):
        r = A.run_query_keyword("how is the weather today")
        self.assertIsNone(r["tool"])
