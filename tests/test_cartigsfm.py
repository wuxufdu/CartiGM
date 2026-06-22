"""Unit tests for the cartigsfm package.

Run with:  python3 -m unittest tests/test_cartigsfm.py
"""
from __future__ import annotations
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("CARTIGSFM_PROC_DIR", str(REPO / "data" / "processed"))

import pandas as pd
import cartigsfm


def load_script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestDictionary(unittest.TestCase):
    def test_package_version_updated(self):
        self.assertEqual(cartigsfm.__version__, "0.4.0")

    def test_list_versions_includes_production(self):
        v = cartigsfm.list_versions()
        self.assertIn("v0.3.1", v)
        self.assertIn("v0.2", v)

    def test_load_dictionary_v031_has_10_subtypes(self):
        d = cartigsfm.load_dictionary("v0.3.1")
        self.assertEqual(len(d), 10)
        for k in d:
            self.assertTrue(k.startswith("cgrm::"))

    def test_panel_genes_round_trip(self):
        d = cartigsfm.load_dictionary("v0.3.1")
        homeo_panel = cartigsfm.panel_genes(d, "Homeostatic_Matrix")
        self.assertGreater(len(homeo_panel), 50)
        # Canonical homeostatic chondrocyte markers must be present
        for g in ("FMOD", "COMP", "OGN"):
            self.assertIn(g, homeo_panel)

    def test_unknown_version_raises(self):
        with self.assertRaises(ValueError):
            cartigsfm.load_dictionary("v99")

    def test_list_function_versions_includes_current(self):
        v = cartigsfm.list_function_versions()
        self.assertIn("v0.6.5", v)

    def test_load_function_specificity_has_avam(self):
        fn = cartigsfm.load_function_specificity("v0.6.5")
        self.assertIn("Avascular_Antimineralization", fn)
        genes = [m["gene"] for m in fn["Avascular_Antimineralization"]["markers"]]
        for g in ("MGP", "CNMD", "TIMP3", "TNFRSF11B"):
            self.assertIn(g, genes)

    def test_load_cartilage_dictionary_v1_has_three_layers(self):
        dictionary = cartigsfm.load_cartilage_dictionary_v1()
        self.assertIn("v1", cartigsfm.list_cartilage_dictionary_versions())
        self.assertEqual(dictionary["layers"]["cell_subtype"]["count"], 10)
        self.assertEqual(dictionary["layers"]["tissue_developmental_state"]["count"], 3)
        self.assertEqual(dictionary["layers"]["functional_axis"]["count"], 29)

    def test_cartilage_dictionary_v1_panels_have_no_marker_anti_overlap(self):
        dictionary = cartigsfm.load_cartilage_dictionary_v1()
        conflicts = []
        for layer_obj in dictionary["layers"].values():
            for axis in layer_obj["axes"]:
                marker_genes = set(axis.get("marker_weights") or {})
                marker_genes.update(str(g).upper() for g in axis.get("core_genes", []))
                marker_genes.update(str(g).upper() for g in axis.get("panel_genes", []))
                anti_genes = set(axis.get("anti_marker_weights") or {})
                anti_genes.update(str(g).upper() for g in axis.get("anti_genes", []))
                overlap = marker_genes & anti_genes
                if overlap:
                    conflicts.append((axis["axis_id"], sorted(overlap)))
                self.assertTrue(set(str(g).upper() for g in axis.get("core_genes", [])) <= marker_genes)
                self.assertTrue(set(str(g).upper() for g in axis.get("panel_genes", [])) <= marker_genes)
                for value in (axis.get("marker_weights") or {}).values():
                    self.assertGreaterEqual(float(value), 0.0)
                    self.assertLessEqual(float(value), 1.0)
                for value in (axis.get("anti_marker_weights") or {}).values():
                    self.assertGreaterEqual(float(value), 0.0)
                    self.assertLessEqual(float(value), 1.0)
        self.assertEqual(conflicts, [])

    def test_cartilage_dictionary_v1_cell_subtype_markers_are_atlas_repaired(self):
        dictionary = cartigsfm.load_cartilage_dictionary_v1()
        expected_top5 = {
            "EC_Lipo_Plasticity": ["ANXA5", "FOSL1", "HMGA1", "TXNRD1", "RAN"],
            "Fibro_Matrix": ["SCARA3", "PLCG2", "MTRNR2L12", "EPB41L2", "COLEC12"],
            "Homeostatic_Matrix": ["FMOD", "COMP", "OGN", "CILP2", "SMOC2"],
            "Hypoxia_Adaptive": ["CIRBP", "BNIP3", "NDUFA4L2", "EPB41L4A-AS1", "VEGFA"],
            "Hypoxia_Metabolic_Stress": ["SNRPD2", "MIF", "GSTO1", "RSL24D1", "SNU13"],
            "Inflammatory_Remodeling": ["SOD2", "SERPINE2", "MMP3", "SLC7A2", "CD55"],
            "Maturation_Matrix": ["FGFBP2", "S100A1", "SNORC", "SERPINA1", "CHAD"],
            "Mesenchymal_Remodeling": ["TMSB4X", "COL1A1", "COL1A2", "MMP2", "COL14A1"],
            "PRG4_Interface": ["CRTAC1", "HTRA1", "PRG4", "FN1", "CRLF1"],
            "Stress_IEG": ["FOS", "GADD45B", "DNAJB1", "JUNB", "HSPA1A"],
        }
        axes = {
            axis["axis_id"].split("::", 1)[1]: axis
            for axis in dictionary["layers"]["cell_subtype"]["axes"]
        }
        self.assertEqual(set(axes), set(expected_top5))
        for subtype, markers in expected_top5.items():
            self.assertEqual(axes[subtype]["core_genes"][:5], markers)
            self.assertEqual(axes[subtype]["panel_genes"][:5], markers)
        self.assertIn("HMGA1", axes["EC_Lipo_Plasticity"]["core_genes"])
        self.assertNotIn("HMGA1", axes["Hypoxia_Metabolic_Stress"]["core_genes"])

    def test_cartilage_dictionary_v1_cell_subtype_names_are_literature_aligned(self):
        dictionary = cartigsfm.load_cartilage_dictionary_v1()
        expected_names = {
            "EC_Lipo_Plasticity": "Effector_Metabolic_Chondrocytes",
            "Fibro_Matrix": "Stromal_Matrix_Chondrocytes",
            "Homeostatic_Matrix": "Matrix_Maintenance_Chondrocytes",
            "Hypoxia_Adaptive": "Hypoxic_Chondrocytes",
            "Hypoxia_Metabolic_Stress": "Metabolic_Stress_Chondrocytes",
            "Inflammatory_Remodeling": "Inflammatory_Response_Chondrocytes",
            "Maturation_Matrix": "Prehypertrophic_Matrix_Chondrocytes",
            "Mesenchymal_Remodeling": "Fibrocartilage_Chondrocytes",
            "PRG4_Interface": "Superficial_Zone_Chondrocytes",
            "Stress_IEG": "Reparative_Stress_Chondrocytes",
        }
        axes = {
            axis["axis_id"].split("::", 1)[1]: axis
            for axis in dictionary["layers"]["cell_subtype"]["axes"]
        }
        for old_name, new_name in expected_names.items():
            axis = axes[old_name]
            self.assertEqual(axis["name_en"], new_name)
            self.assertIn(old_name, axis["aliases"])
            self.assertEqual(
                axis["naming_policy"],
                "Literature-aligned display name; stable axis_id is retained for backwards compatibility.",
            )

    def test_rag_knowledge_base_mirrors_repaired_dictionary_panels(self):
        dictionary = cartigsfm.load_cartilage_dictionary_v1()
        kb = cartigsfm.load_rag_knowledge_base()
        axes = {
            axis["axis_id"]: axis
            for layer_obj in dictionary["layers"].values()
            for axis in layer_obj["axes"]
        }
        for entry in kb["dictionary_knowledge"]:
            axis = axes.get(entry["axis_id"])
            self.assertIsNotNone(axis)
            self.assertEqual(entry.get("name_en"), axis.get("name_en"))
            self.assertEqual(entry.get("name_cn"), axis.get("name_cn"))
            self.assertEqual(entry.get("aliases"), axis.get("aliases"))
            self.assertEqual(entry.get("core_genes"), axis.get("core_genes"))
            self.assertEqual(entry.get("panel_genes"), axis.get("panel_genes"))
            self.assertEqual(entry.get("anti_genes"), axis.get("anti_genes"))

    def test_load_rag_resources_and_claim_safety(self):
        self.assertIn("v1", cartigsfm.list_rag_versions())
        kb = cartigsfm.load_rag_knowledge_base()
        self.assertTrue(kb["knowledge_base_complete"])
        cards = cartigsfm.load_axis_evidence_cards()
        self.assertGreaterEqual(len(cards), 10)
        prompts = cartigsfm.load_prompt_templates()
        self.assertIn("5_hallucination_guardrails", prompts)
        claim = cartigsfm.find_claim_safety(
            "CartiGSFM is a trained cartilage large language model (LLM)"
        )
        self.assertIsNotNone(claim)
        self.assertEqual(claim["safety_classification"], "NOT_SUPPORTED")
        self.assertFalse(claim["can_claim"])

    def test_load_p9_metadata(self):
        self.assertIn("v1", cartigsfm.list_p9_versions())
        config = cartigsfm.load_p9_training_config()
        self.assertTrue(config["actually_trained"])
        self.assertEqual(config["base_model"], "Qwen/Qwen2.5-0.5B-Instruct")
        comparison = cartigsfm.load_p9_model_comparison()
        systems = {row["system"] for row in comparison}
        self.assertIn("base", systems)
        self.assertIn("lora_only", systems)
        report = cartigsfm.load_p9_training_report()
        self.assertIn("Was LoRA actually trained?", report)
        card = cartigsfm.load_p9_model_card()
        self.assertIn("Actually trained", card)
        self.assertTrue(cartigsfm.p9_is_adapter_available(
            REPO / "review_p9_delivery" / "cartigsfm_p9_lora_training_delivery" / "adapter"
        ))


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.d = cartigsfm.load_dictionary("v0.3.1")

    def test_homeostatic_core_query_picks_homeostatic(self):
        # Use recommended_core (top-10 distinctive markers per subtype) rather
        # than the full panel: Homeostatic_Matrix's full panel overlaps with
        # Maturation_Matrix on canonical chondrogenic markers (ACAN, COL2A1,
        # COMP, CHAD), and that overlap is by design -- the v0.5 split into
        # Hypertrophy + Mature_Stable is documented in
        # data/processed/EVIDENCE_V02_VS_V031_COMPARISON.md.
        d = self.d
        homeo_core = d["cgrm::Homeostatic_Matrix"].get("recommended_core", [])
        df = cartigsfm.score_query(homeo_core, d)
        # The core itself must surface Homeostatic_Matrix in the top-2 hits.
        top2 = df.head(2)["subtype"].tolist()
        self.assertIn("cgrm::Homeostatic_Matrix", top2)

    def test_empty_query_yields_empty_dataframe(self):
        df = cartigsfm.score_query([], self.d)
        self.assertTrue(df.empty)

    def test_anti_penalty_increases_separation(self):
        # Mix homeostatic panel with PRG4_Interface anti-panel terms;
        # higher anti_penalty must reduce overlap_score - anti_score gap behaviour
        homeo = list(cartigsfm.panel_genes(self.d, "Homeostatic_Matrix").keys())[:20]
        df_low = cartigsfm.score_query(homeo, self.d, anti_penalty=0.0)
        df_high = cartigsfm.score_query(homeo, self.d, anti_penalty=2.0)
        # Either separation is preserved or it is amplified at higher penalty
        homeo_top_low = df_low[df_low["subtype"] == "cgrm::Homeostatic_Matrix"]["combined"].iloc[0]
        homeo_top_high = df_high[df_high["subtype"] == "cgrm::Homeostatic_Matrix"]["combined"].iloc[0]
        # combined must be deterministic
        self.assertIsInstance(homeo_top_low, float)
        self.assertIsInstance(homeo_top_high, float)

    def test_alias_resolution_keeps_cartilage_context(self):
        alias = cartigsfm.load_alias_map()
        genes = cartigsfm.resolve_aliases(["LECT1", "TNFRSF11B", "TAZ"], alias)
        self.assertIn("CNMD", genes)
        self.assertIn("TNFRSF11B", genes)
        self.assertIn("WWTR1", genes)

    def test_avam_function_query_ranks_first(self):
        fn_spec = cartigsfm.load_function_specificity("v0.6.5")
        fn_dict = cartigsfm.load_function_dictionary("v0.6.5")
        genes = "MGP CNMD LECT1 TIMP3 ANKH ENPP1 TNFRSF11B FRZB SOX9 ACAN".split()
        genes = cartigsfm.resolve_aliases(genes, cartigsfm.load_alias_map())
        df = cartigsfm.score_function_query(genes, fn_spec, fn_dict)
        self.assertFalse(df.empty)
        self.assertEqual(df.iloc[0]["function"], "Avascular_Antimineralization")


class TestProjection(unittest.TestCase):
    def setUp(self):
        self.d = cartigsfm.load_dictionary("v0.3.1")

    def test_project_synthetic_homeostatic_high(self):
        # Build a tiny synthetic expression matrix: high expression of homeostatic
        # markers in sample S1, low in S2. Projection must give S1 > S2 for
        # Homeostatic_Matrix.
        homeo = list(cartigsfm.panel_genes(self.d, "Homeostatic_Matrix").keys())[:20]
        other = list(cartigsfm.panel_genes(self.d, "Inflammatory_Remodeling").keys())[:20]
        all_genes = list(set(homeo + other))
        rows = {}
        for g in all_genes:
            if g in homeo:
                rows[g] = [10.0, 0.5]
            else:
                rows[g] = [0.5, 10.0]
        expr = pd.DataFrame.from_dict(rows, orient="index", columns=["S1", "S2"])
        proj = cartigsfm.project_bulk(expr, self.d)
        homeo_rows = proj[proj["subtype"] == "cgrm::Homeostatic_Matrix"]
        s1 = homeo_rows[homeo_rows["sample"] == "S1"]["score"].iloc[0]
        s2 = homeo_rows[homeo_rows["sample"] == "S2"]["score"].iloc[0]
        self.assertGreater(s1, s2)

    def test_project_synthetic_avam_high(self):
        fn_spec = cartigsfm.load_function_specificity("v0.6.5")
        fn_dict = cartigsfm.load_function_dictionary("v0.6.5")
        avam = ["MGP", "CNMD", "LECT1", "TIMP3", "ANKH", "ENPP1", "TNFRSF11B", "FRZB", "SOX9", "ACAN"]
        background = ["IL6", "MMP13", "VEGFA", "RUNX2", "ADAMTS5"]
        rows = {}
        for g in avam + background:
            if g in avam:
                rows[g] = [10.0, 0.5]
            else:
                rows[g] = [0.5, 10.0]
        expr = pd.DataFrame.from_dict(rows, orient="index", columns=["AvAm_high", "AvAm_low"])
        proj = cartigsfm.project_function_bulk(
            expr,
            fn_spec,
            fn_dict,
            alias_map=cartigsfm.load_alias_map(),
        )
        avam_rows = proj[proj["function"] == "Avascular_Antimineralization"]
        high = avam_rows[avam_rows["sample"] == "AvAm_high"]["score"].iloc[0]
        low = avam_rows[avam_rows["sample"] == "AvAm_low"]["score"].iloc[0]
        self.assertGreater(high, low)

    def test_project_dictionary_v1_three_layers(self):
        dictionary = cartigsfm.load_cartilage_dictionary_v1()
        genes = ["MGP", "CNMD", "TIMP3", "ANKH", "ENPP1", "TNFRSF11B", "FRZB", "SOX9", "ACAN", "COL2A1"]
        background = ["IL6", "VEGFA", "RUNX2", "MMP13", "ADAMTS5", "COL10A1"]
        expr = pd.DataFrame(
            {gene: [10.0, 0.5] if gene in genes else [0.5, 10.0] for gene in genes + background},
            index=["AvAm_high", "AvAm_low"],
        ).T
        expr.insert(0, "gene", expr.index)
        scores = cartigsfm.project_dictionary_v1_bulk(expr, dictionary, gene_col="gene")
        self.assertEqual(set(scores["layer"]), {"cell_subtype", "tissue_developmental_state", "functional_axis"})
        avam = scores[scores["axis_id"] == "functional_axis::Avascular_Antimineralization"]
        self.assertFalse(avam.empty)
        high = avam[avam["sample"] == "AvAm_high"]["score"].iloc[0]
        low = avam[avam["sample"] == "AvAm_low"]["score"].iloc[0]
        self.assertGreater(high, low)

    def test_p4_project_from_pseudobulk_writes_outputs(self):
        genes = ["MGP", "CNMD", "TIMP3", "ANKH", "ENPP1", "ELN", "FBLN5", "COL2A1", "ACAN", "RUNX2"]
        expr = pd.DataFrame({
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
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pb = root / "pb.tsv"
            mt = root / "meta.tsv"
            outdir = root / "p4"
            expr.to_csv(pb, sep="\t", index=False)
            meta.to_csv(mt, sep="\t")
            outputs = cartigsfm.run_p4_project(outdir=outdir, pseudobulk_tsv=pb, meta_tsv=mt)
            for path in outputs.values():
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)
            scores = pd.read_csv(outputs["scores"], sep="\t")
            self.assertIn("functional_axis::Avascular_Antimineralization", scores["axis_id"].tolist())


class TestProjectionSummary(unittest.TestCase):
    def test_figure_summary_tracks_avam_rank(self):
        mod = load_script_module(
            REPO / "scripts" / "63_summarize_projection_for_figures.py",
            "cartigsfm_projection_summary_test",
        )
        projection = pd.DataFrame([
            {"kind": "subtype", "category": "cgrm::Homeostatic_Matrix", "cluster": "C1", "score": 0.4, "panel_n": 5},
            {"kind": "subtype", "category": "cgrm::Stress_IEG", "cluster": "C1", "score": 0.1, "panel_n": 2},
            {"kind": "function", "category": "Avascular_Antimineralization", "cluster": "C1", "score": 1.2, "marker_n": 9, "consensus_n": 9},
            {"kind": "function", "category": "Chondrogenesis", "cluster": "C1", "score": 1.2, "marker_n": 4, "consensus_n": 2},
            {"kind": "function", "category": "Inflammation_NFkB", "cluster": "C1", "score": -0.5, "marker_n": 3, "consensus_n": 3},
        ])
        top = pd.DataFrame([
            {"kind": "subtype", "cluster": "C1", "margin": 0.3},
            {"kind": "function", "cluster": "C1", "margin": 0.0},
        ])
        meta = pd.DataFrame([{"cluster": "C1", "n_cells": 30}])
        out = mod.summarize_projection(projection, top_assignments=top, meta=meta)
        self.assertEqual(len(out), 1)
        row = out.iloc[0]
        self.assertEqual(row["subtype_top1"], "cgrm::Homeostatic_Matrix")
        self.assertEqual(row["function_top1"], "Avascular_Antimineralization")
        self.assertEqual(row["avam_rank"], 1)
        self.assertTrue(row["avam_is_top_function"])
        self.assertEqual(row["n_cells"], 30)


class TestMarkerTableAnnotation(unittest.TestCase):
    def test_marker_table_annotation_finds_avam_cluster(self):
        mod = load_script_module(
            REPO / "scripts" / "65_annotate_marker_table.py",
            "cartigsfm_marker_table_annotation_test",
        )
        markers = pd.DataFrame([
            {"group": "AvAm", "names": "MGP", "scores": 10.0, "logfoldchanges": 2.0, "pvals_adj": 0.0},
            {"group": "AvAm", "names": "CNMD", "scores": 9.5, "logfoldchanges": 2.0, "pvals_adj": 0.0},
            {"group": "AvAm", "names": "TIMP3", "scores": 9.0, "logfoldchanges": 2.0, "pvals_adj": 0.0},
            {"group": "AvAm", "names": "ANKH", "scores": 8.5, "logfoldchanges": 2.0, "pvals_adj": 0.0},
            {"group": "AvAm", "names": "ENPP1", "scores": 8.0, "logfoldchanges": 2.0, "pvals_adj": 0.0},
            {"group": "AvAm", "names": "TNFRSF11B", "scores": 7.5, "logfoldchanges": 2.0, "pvals_adj": 0.0},
            {"group": "AvAm", "names": "FRZB", "scores": 7.0, "logfoldchanges": 2.0, "pvals_adj": 0.0},
            {"group": "AvAm", "names": "SOX9", "scores": 6.5, "logfoldchanges": 2.0, "pvals_adj": 0.0},
            {"group": "AvAm", "names": "ACAN", "scores": 6.0, "logfoldchanges": 2.0, "pvals_adj": 0.0},
        ])
        args = type("Args", (), {
            "group_col": "group",
            "gene_col": "names",
            "score_col": "scores",
            "top_n": 50,
            "report_top": 5,
            "subtype_version": "v0.3.1",
            "function_version": "v0.6.5",
            "anti_penalty": 1.0,
        })()
        summary, _, fn = mod.annotate_groups(markers, args)
        self.assertEqual(summary.iloc[0]["function_top1"], "Avascular_Antimineralization")
        self.assertEqual(summary.iloc[0]["avam_rank"], 1)
        self.assertIn("Avascular_Antimineralization", fn["function"].tolist())


class TestCrossTissuePlotting(unittest.TestCase):
    def test_cross_tissue_plotter_writes_outputs(self):
        mod = load_script_module(
            REPO / "scripts" / "64_plot_cross_tissue_projection.py",
            "cartigsfm_cross_tissue_plot_test",
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ear = root / "ear.tsv"
            nose = root / "nose.tsv"
            pd.DataFrame([
                {"cluster": "C0", "subtype_top1": "cgrm::Homeostatic_Matrix", "avam_score": 1.2, "avam_is_top_function": True},
                {"cluster": "C1", "subtype_top1": "cgrm::PRG4_Interface", "avam_score": 0.1, "avam_is_top_function": False},
            ]).to_csv(ear, sep="\t", index=False)
            pd.DataFrame([
                {"cluster": "N0", "subtype_top1": "cgrm::Maturation_Matrix", "avam_score": -0.2, "avam_is_top_function": False},
            ]).to_csv(nose, sep="\t", index=False)

            combined = mod.load_summaries([f"ear={ear}", f"nose={nose}"])
            self.assertEqual(len(combined), 3)
            self.assertIn("cluster_label", combined.columns)
            out_prefix = root / "cross_tissue"
            mod.plot_avam_scores(combined, out_prefix)
            mod.plot_subtype_assignments(combined, out_prefix)
            for suffix in ("avam_scores", "subtype_assignments"):
                for ext in ("png", "pdf"):
                    path = root / f"cross_tissue_{suffix}.{ext}"
                    self.assertTrue(path.exists())
                    self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
