"""Smoke test for cs_classifier loader; uses the bundled checkpoint when present."""
from __future__ import annotations

import unittest
from pathlib import Path


class TestCSClassifierLoader(unittest.TestCase):
    def _load(self):
        from cartigsfm.cs_classifier import bundled_classifier_path, load_classifier
        ckpt = bundled_classifier_path()
        if not ckpt.exists():
            self.skipTest(f"bundled classifier checkpoint missing at {ckpt}")
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch unavailable")
        return load_classifier(ckpt, device="cpu")

    def test_bundled_checkpoint_loads(self) -> None:
        model, classes, genes, cfg = self._load()
        self.assertEqual(len(classes), cfg.n_classes)
        self.assertEqual(len(genes), cfg.n_in)
        self.assertEqual(cfg.n_classes, 10)
        # Classes should match the cartilage_dictionary_v1 cell_subtype panel
        # (10 axes, in any order).
        for c in classes:
            self.assertIn("Chondrocytes", c)

    def test_align_and_predict_zero(self) -> None:
        import numpy as np
        from cartigsfm.cs_classifier import align_to_genes, predict_from_array
        model, classes, genes, cfg = self._load()
        # zero matrix -> still produces a softmax over n_classes
        X = np.zeros((4, len(genes)), dtype=np.float32)
        idx, probs = predict_from_array(X, model, classes, device="cpu")
        self.assertEqual(idx.shape, (4,))
        self.assertEqual(probs.shape, (4, len(classes)))
        # probabilities should sum to 1 per cell
        for row_sum in probs.sum(axis=1):
            self.assertAlmostEqual(float(row_sum), 1.0, places=4)

        # align_to_genes round-trip with a partial src panel
        src = list(genes[:50]) + ["FAKE_GENE_NOT_PRESENT"]
        Xs = np.ones((3, len(src)), dtype=np.float32)
        Xa, hit = align_to_genes(Xs, src, genes)
        self.assertEqual(hit, 50)
        self.assertEqual(Xa.shape, (3, len(genes)))
        self.assertTrue(np.all(Xa[:, :50] == 1.0))
        self.assertTrue(np.all(Xa[:, 50:] == 0.0))


if __name__ == "__main__":
    unittest.main()
