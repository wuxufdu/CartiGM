from __future__ import annotations

import unittest


try:
    from fastapi.testclient import TestClient
    from cartigsfm_web.server import create_app
except Exception as exc:  # pragma: no cover - dependency skip path
    TestClient = None
    create_app = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(TestClient is None, f"FastAPI test dependencies unavailable: {IMPORT_ERROR}")
class TestCartiGMWebPortal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app())

    def test_health_reports_dictionary(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["dictionary_version"], "v1.8.6")
        self.assertGreaterEqual(data["axis_count"], 53)

    def test_dictionary_endpoint_contains_three_layers(self):
        response = self.client.get("/api/dictionary")
        self.assertEqual(response.status_code, 200)
        layers = response.json()["layers"]
        self.assertIn("cell_subtype", layers)
        self.assertIn("tissue_developmental_state", layers)
        self.assertIn("functional_axis", layers)
        self.assertEqual(layers["cell_subtype"]["count"], 10)
        self.assertEqual(layers["tissue_developmental_state"]["count"], 4)
        self.assertEqual(layers["functional_axis"]["count"], 39)

    def test_score_endpoint_returns_evidence_constrained_hits(self):
        response = self.client.post(
            "/api/score",
            json={"genes": "COL2A1, ACAN, SOX9, PRG4, COL10A1", "top": 3},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["n_input_genes"], 5)
        self.assertIn("safety_note", data)
        self.assertIn("by_layer", data)
        self.assertTrue(data["overall"])
        top = data["overall"][0]
        self.assertIn("marker_hits", top)
        self.assertIn("limitations", top)

    def test_empty_score_request_is_rejected(self):
        response = self.client.post("/api/score", json={"genes": "   "})
        self.assertEqual(response.status_code, 400)

    def test_claim_check_endpoint(self):
        response = self.client.post(
            "/api/claim-check",
            json={"claim": "CartiGM is a trained cartilage large language model (LLM)"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("matched", data)
        self.assertIn("recommendation", data)

    def test_interpret_endpoint_returns_markdown_and_claim_safety(self):
        response = self.client.post(
            "/api/interpret",
            json={
                "genes": "MGP,CNMD,TIMP3,ANKH,ENPP1,TNFRSF11B,FRZB,SOX9,ACAN",
                "top_per_layer": 2,
                "overall_top": 5,
                "claims": "CartiGM is a trained cartilage large language model (LLM)",
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("markdown", data)
        self.assertIn("CartiGSFM Evidence-Constrained Interpretation", data["markdown"])
        self.assertIn("safety_summary", data)
        self.assertTrue(data["top_axes_per_layer"])
        self.assertTrue(data["cannot_claim"])

    def test_p4_command_builder_does_not_upload_h5ad(self):
        response = self.client.post(
            "/api/p4-command",
            json={
                "h5ad_path": "EBR.h5ad",
                "outdir": "P4_EBR",
                "sample_col": "sample",
                "tissue_col": "tissue",
                "cluster_col": "leiden",
                "layer": "log1p_norm",
                "min_cells": 10,
                "streaming": "force",
                "chunk_size": 10000,
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('--h5ad "EBR.h5ad"', data["command"])
        self.assertIn("--layer log1p_norm", data["command"])
        self.assertIn("--min-cells 10", data["command"])
        self.assertIn("--streaming", data["command"])
        self.assertIn("--chunk-size 10000", data["command"])
        self.assertIn("does not upload", data["note"])

    def test_inspect_command_builder(self):
        response = self.client.post(
            "/api/inspect-command",
            json={"h5ad_path": "acc_new.h5ad", "output_format": "json"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("inspect-h5ad", data["command"])
        self.assertIn('--h5ad "acc_new.h5ad"', data["command"])
        self.assertIn("--format json", data["command"])
        self.assertIn("Run this locally", data["note"])

    def test_cs_predict_command_builder(self):
        response = self.client.post(
            "/api/cs-predict-command",
            json={
                "h5ad_path": "EBR.h5ad",
                "out": "preds.tsv",
                "mode": "ensemble",
                "layer": "log1p_norm",
                "device": "cuda",
                "batch_size": 2048,
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("cs-predict", data["command"])
        self.assertIn('--out "preds.tsv"', data["command"])
        self.assertIn("--mode ensemble", data["command"])
        self.assertIn("--layer log1p_norm", data["command"])
        self.assertIn("--device cuda", data["command"])
        self.assertIn("--batch-size 2048", data["command"])


if __name__ == "__main__":
    unittest.main()
