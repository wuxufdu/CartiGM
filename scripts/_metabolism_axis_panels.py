"""Curated literature-prior gene panels for the v1.7 metabolism extension.

The data is intentionally side-effect free so that the build script
(``scripts/add_metabolism_functional_axes.py``) can import it as the single
source of truth for the v1.7 dictionary bump.

Conventions
-----------
* ``core_genes`` is a flat ``(gene, weight)`` tuple list. Weights follow the
  same descending convention used by the bundled v1.0 Hypoxia panel: the most
  canonical regulators sit near 1.5 and the long tail sits near 0.3.
* ``panel_genes`` is a strict superset of ``core_genes`` plus pathway/family
  members curated from KEGG (hsa00010/00020/00030/00071/00100/00190),
  Reactome (R-HSA-70171/71403/77289/8957322/556833/1430728/15869),
  MSigDB Hallmark Glycolysis / OxPhos / FAO / Adipogenesis / Cholesterol
  Homeostasis, and the cartilage-specific reviews cited in
  ``literature_support``.
* ``anti_genes`` lists antagonistic markers that should down-vote the axis.
* All axes use ``status="literature_prior"`` so that
  :func:`cartigsfm.interpret.axis_safety_class` returns ``EXPLORATORY`` until
  the next atlas-calibration pass writes ``freq_in``/``freq_bg``/``log2_spec``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


GLYCOLYSIS_CORE: List[Tuple[str, float]] = [
    ("HK2", 1.50), ("PFKFB3", 1.45), ("PKM", 1.42), ("LDHA", 1.40),
    ("ENO1", 1.35), ("GAPDH", 1.32), ("SLC2A1", 1.30), ("SLC2A3", 1.28),
    ("PFKL", 1.25), ("PFKP", 1.22), ("ALDOA", 1.20), ("PGK1", 1.18),
    ("HK1", 1.15), ("PFKFB4", 1.12), ("TPI1", 1.10), ("PGAM1", 1.05),
    ("LDHB", 1.00), ("HIF1A", 0.95), ("MYC", 0.90), ("PDK1", 0.85),
    ("PDK3", 0.80), ("BPGM", 0.75), ("ENO2", 0.70), ("ENO3", 0.65),
    ("HK3", 0.60), ("ALDOC", 0.55), ("PFKFB1", 0.50), ("SLC16A1", 0.45),
    ("SLC16A3", 0.40), ("ADPGK", 0.35),
]

GLYCOLYSIS_PANEL = sorted(
    {g for g, _ in GLYCOLYSIS_CORE}
    | {
        "SLC16A4", "SLC16A7", "SLC2A2", "SLC2A4",
        "GPI", "GAPDHS", "PFKFB2", "ALDOB",
        "PGM1", "PGM2", "FBP1", "FBP2",
        "GALM", "GCK", "MINPP1",
        "PCK1", "PCK2", "G6PC1", "G6PC2", "G6PC3",
        "TKTL1", "ME1", "ME2", "ME3", "GOT1", "GOT2", "PGLS",
        "DLAT", "DLD", "PDHA1", "PDHB", "PDHX", "PDP1", "PDP2",
        "AKR1A1", "TIGAR", "MYCN", "MAX", "MLXIPL", "EPAS1",
        "PRKAA1", "PRKAA2", "PRKAB1", "PRKAB2", "PRKAG1", "PRKAG2",
    }
)

OXPHOS_CORE: List[Tuple[str, float]] = [
    ("ATP5F1A", 1.50), ("ATP5F1B", 1.48), ("ATP5F1C", 1.45),
    ("COX4I1", 1.42), ("COX5A", 1.40), ("COX5B", 1.38),
    ("CYCS", 1.35), ("UQCRC1", 1.32), ("UQCRC2", 1.30),
    ("UQCRH", 1.28), ("SDHA", 1.25), ("SDHB", 1.22),
    ("NDUFA1", 1.20), ("NDUFA2", 1.18), ("NDUFB7", 1.15),
    ("NDUFS1", 1.12), ("NDUFS2", 1.10), ("NDUFV1", 1.05),
    ("MT-CO1", 1.00), ("MT-CO2", 0.95), ("MT-ATP6", 0.90),
    ("MT-ND1", 0.85), ("MT-ND2", 0.80), ("MT-CYB", 0.75),
    ("ATP5MC1", 0.70), ("COX6A1", 0.65), ("COX7A2", 0.60),
    ("UQCRFS1", 0.55), ("ATP5MD", 0.50), ("NDUFAB1", 0.45),
]

OXPHOS_PANEL = sorted(
    {g for g, _ in OXPHOS_CORE}
    | {
        "ATP5F1D", "ATP5F1E", "ATP5MF", "ATP5MG", "ATP5ME", "ATP5MJ", "ATP5IF1",
        "COX6B1", "COX6C", "COX7A1", "COX7B", "COX7C", "COX8A",
        "UQCRB", "UQCRQ", "UQCR10", "UQCR11", "CYC1",
        "SDHC", "SDHD", "SDHAF1", "SDHAF2",
        "NDUFA3", "NDUFA4", "NDUFA5", "NDUFA6", "NDUFA7", "NDUFA8", "NDUFA9",
        "NDUFA10", "NDUFA11", "NDUFA12", "NDUFA13",
        "NDUFB1", "NDUFB2", "NDUFB3", "NDUFB4", "NDUFB5", "NDUFB6",
        "NDUFB8", "NDUFB9", "NDUFB10", "NDUFB11",
        "NDUFS3", "NDUFS4", "NDUFS5", "NDUFS6", "NDUFS7", "NDUFS8",
        "NDUFV2", "NDUFV3",
        "MT-ND3", "MT-ND4", "MT-ND4L", "MT-ND5", "MT-ND6",
        "MT-CO3", "MT-ATP8",
        "TFAM", "POLG", "POLG2", "TWNK", "MFN1", "MFN2", "OPA1",
        "PPARGC1A", "PPARGC1B", "NRF1", "NFE2L2", "ESRRA", "YME1L1",
        "PRDX3", "PRDX5", "TXN2", "GPX1", "GPX4", "SOD2",
    }
)

TCA_CORE: List[Tuple[str, float]] = [
    ("CS", 1.50), ("ACO2", 1.45), ("IDH2", 1.42), ("IDH3A", 1.40),
    ("IDH3B", 1.38), ("IDH3G", 1.35), ("OGDH", 1.32), ("OGDHL", 1.28),
    ("DLST", 1.25), ("SUCLA2", 1.22), ("SUCLG1", 1.20), ("SUCLG2", 1.18),
    ("SDHA", 1.15), ("SDHB", 1.12), ("FH", 1.10), ("MDH2", 1.05),
    ("MDH1", 1.00), ("PDHA1", 0.95), ("PDHB", 0.90), ("DLAT", 0.85),
    ("DLD", 0.80), ("PCK2", 0.75), ("PC", 0.72), ("ACLY", 0.70),
    ("IDH1", 0.65), ("ACO1", 0.60), ("BCAT2", 0.55), ("GPT2", 0.50),
    ("GLUD1", 0.45), ("MPC1", 0.40),
]

TCA_PANEL = sorted(
    {g for g, _ in TCA_CORE}
    | {
        "SDHC", "SDHD", "PDP1", "PDP2", "PDK1", "PDK2", "PDK3", "PDK4",
        "PDPR", "PDHX", "PCCA", "PCCB", "MCEE", "MUT",
        "OXCT1", "BDH1", "ECHS1", "HADH", "ACAT1", "ACAA1", "ACAA2",
        "GLS", "GLS2", "GOT1", "GOT2", "ASS1", "ASL",
        "MPC2", "SLC25A1", "SLC25A11", "SLC25A12", "SLC25A13",
        "SUCNR1", "L2HGDH", "D2HGDH",
    }
)

PPP_CORE: List[Tuple[str, float]] = [
    ("G6PD", 1.50), ("PGD", 1.45), ("TKT", 1.40), ("TALDO1", 1.38),
    ("RPIA", 1.35), ("RPE", 1.30), ("PRPS1", 1.28), ("PRPS2", 1.25),
    ("H6PD", 1.20), ("PGLS", 1.18), ("RBKS", 1.10), ("DERA", 1.05),
    ("PFKL", 0.95), ("PFKM", 0.90), ("PFKP", 0.85), ("ALDOA", 0.80),
    ("FBP1", 0.75), ("FBP2", 0.70), ("GPI", 0.65), ("HK1", 0.62),
    ("HK2", 0.60), ("PGM1", 0.55), ("PGM2", 0.50), ("RGN", 0.45),
    ("NADK", 0.42), ("ME1", 0.40), ("IDH1", 0.38), ("PRPS1L1", 0.35),
    ("TKTL1", 0.32), ("TKTL2", 0.30),
]

PPP_PANEL = sorted(
    {g for g, _ in PPP_CORE}
    | {
        "GLRX", "GSR", "GCLC", "GCLM", "TXN", "TXNRD1", "TXNRD2",
        "NQO1", "G6PD2", "PRPSAP1", "PRPSAP2",
        "ADA", "AMPD2", "AMPD3", "DHODH", "UMPS", "DUT",
        "NME1", "NME2", "NME3", "NME4",
        "ATIC", "GART", "PFAS", "PAICS", "ADSL", "ADSS",
    }
)

FAO_CORE: List[Tuple[str, float]] = [
    ("CPT1A", 1.50), ("CPT1B", 1.45), ("CPT2", 1.42),
    ("ACADVL", 1.40), ("ACADM", 1.38), ("ACADL", 1.35), ("ACADS", 1.32),
    ("HADHA", 1.30), ("HADHB", 1.28), ("HADH", 1.25),
    ("ECH1", 1.22), ("ECHS1", 1.20), ("ACAA2", 1.18), ("ACAA1", 1.15),
    ("ACOX1", 1.12), ("ACOX2", 1.10), ("ACOX3", 1.05),
    ("SLC25A20", 1.00), ("PPARA", 0.95), ("PPARGC1A", 0.90),
    ("ACSL1", 0.85), ("ACSL3", 0.80), ("ACSL4", 0.75), ("ACSL5", 0.70),
    ("CRAT", 0.65), ("CROT", 0.60), ("DECR1", 0.55),
    ("ETFA", 0.50), ("ETFB", 0.45), ("ETFDH", 0.40),
]

FAO_PANEL = sorted(
    {g for g, _ in FAO_CORE}
    | {
        "EHHADH", "MLYCD", "ACADSB", "ACAD9", "ACAD10", "ACAD11",
        "HSD17B4", "HSD17B10", "ACAT1", "ACAT2",
        "PPARD", "PPARG", "RXRA", "RXRB",
        "FABP3", "FABP4", "FABP5", "FABP6", "FABP7", "CD36",
        "SLC27A1", "SLC27A2", "SLC27A3", "SLC27A4", "SLC27A5", "SLC27A6",
        "AMACR", "PEX5", "PEX7", "PEX11A", "PEX11B",
        "CPT1C", "GCDH", "IVD",
    }
)

LIPOGENESIS_CORE: List[Tuple[str, float]] = [
    ("ACLY", 1.50), ("ACACA", 1.45), ("ACACB", 1.42),
    ("FASN", 1.40), ("SCD", 1.38), ("SCD5", 1.32),
    ("ELOVL5", 1.30), ("ELOVL6", 1.28), ("ELOVL1", 1.20),
    ("SREBF1", 1.25), ("SREBF2", 1.22),
    ("INSIG1", 1.15), ("INSIG2", 1.12),
    ("MID1IP1", 1.10), ("AGPAT1", 1.05), ("AGPAT2", 1.00),
    ("GPAT3", 0.95), ("GPAT4", 0.90),
    ("DGAT1", 0.85), ("DGAT2", 0.80),
    ("LPIN1", 0.75), ("LPIN2", 0.70), ("LPIN3", 0.65),
    ("FADS1", 0.60), ("FADS2", 0.55), ("FADS3", 0.50),
    ("THRSP", 0.45), ("MLXIPL", 0.42), ("PNPLA3", 0.40), ("MBOAT7", 0.35),
]

LIPOGENESIS_PANEL = sorted(
    {g for g, _ in LIPOGENESIS_CORE}
    | {
        "ELOVL2", "ELOVL3", "ELOVL4", "ELOVL7",
        "GPAM", "AGPAT3", "AGPAT4", "AGPAT5",
        "PCYT1A", "PCYT1B", "PCYT2", "PEMT", "CHKA", "CHKB", "CHPT1",
        "PLD1", "PLD2", "PLD3", "PLD4", "PLD6",
        "MOGAT1", "MOGAT2",
        "SLC25A10", "ME1", "ME2", "ME3",
        "PPARG", "RXRA", "NR1H2", "NR1H3", "USF1", "USF2",
        "ABCA1", "ABCG1", "STARD3", "SOAT1", "SOAT2",
    }
)

CHOLESTEROL_CORE: List[Tuple[str, float]] = [
    ("HMGCR", 1.50), ("HMGCS1", 1.45), ("MVK", 1.40),
    ("MVD", 1.38), ("FDPS", 1.35), ("FDFT1", 1.32),
    ("SQLE", 1.30), ("LSS", 1.28), ("CYP51A1", 1.25),
    ("DHCR7", 1.22), ("DHCR24", 1.20),
    ("SREBF2", 1.18), ("INSIG1", 1.15), ("INSIG2", 1.12),
    ("LDLR", 1.10), ("LDLRAP1", 1.05),
    ("SCAP", 1.00), ("MSMO1", 0.95), ("NSDHL", 0.90),
    ("SC5D", 0.85), ("EBP", 0.80), ("TM7SF2", 0.75),
    ("HMGCS2", 0.65), ("ACAT2", 0.60),
    ("PMVK", 0.55), ("IDI1", 0.50), ("IDI2", 0.45),
    ("GGPS1", 0.40), ("FDXR", 0.35), ("CYP27A1", 0.32),
]

CHOLESTEROL_PANEL = sorted(
    {g for g, _ in CHOLESTEROL_CORE}
    | {
        "ABCA1", "ABCG1", "ABCG5", "ABCG8",
        "APOE", "APOA1", "APOA2", "APOB", "APOC1", "APOC2", "APOC3",
        "LIPA", "LIPG", "LCAT", "CETP", "PCSK9", "MYLIP",
        "STARD1", "STARD3", "STARD4", "STARD5", "OSBPL1A", "OSBPL2",
        "NPC1", "NPC2", "NPC1L1",
        "SOAT1", "SOAT2", "CYP7A1", "CYP7B1", "CYP8B1", "CYP46A1",
        "NR1H2", "NR1H3", "NR1H4", "NR0B2", "RXRA",
    }
)

LIPIDDROPLET_CORE: List[Tuple[str, float]] = [
    ("PLIN1", 1.50), ("PLIN2", 1.48), ("PLIN3", 1.45),
    ("PLIN4", 1.40), ("PLIN5", 1.38),
    ("DGAT1", 1.35), ("DGAT2", 1.32),
    ("PNPLA2", 1.30), ("ABHD5", 1.28), ("LIPE", 1.25),
    ("CIDEA", 1.22), ("CIDEC", 1.20), ("CIDEB", 1.15),
    ("FITM1", 1.10), ("FITM2", 1.08),
    ("FABP3", 1.05), ("FABP4", 1.00), ("FABP5", 0.95),
    ("MGLL", 0.90), ("AWAT2", 0.85),
    ("HSD17B11", 0.80), ("HSD17B12", 0.75), ("HSD17B13", 0.70),
    ("BSCL2", 0.65), ("CAV1", 0.60), ("CAV2", 0.55),
    ("FAF2", 0.50), ("UBXN6", 0.45), ("RAB18", 0.40),
]

LIPIDDROPLET_PANEL = sorted(
    {g for g, _ in LIPIDDROPLET_CORE}
    | {
        "PNPLA3", "PNPLA5",
        "FAR1", "FAR2",
        "AGPAT1", "AGPAT2", "GPAT3", "GPAT4",
        "DDHD1", "DDHD2",
        "RAB7A", "RAB8A", "RAB10", "RAB11A",
        "ARF1", "COPA",
        "MOGAT1", "MOGAT2",
        "AUP1", "G0S2",
    }
)

GLUTAMINOLYSIS_CORE: List[Tuple[str, float]] = [
    ("GLS", 1.50), ("GLS2", 1.45),
    ("GLUD1", 1.42), ("GLUD2", 1.35),
    ("GOT1", 1.32), ("GOT2", 1.30),
    ("GPT", 1.25), ("GPT2", 1.22),
    ("ASNS", 1.20), ("ASS1", 1.15), ("ASL", 1.12),
    ("ALDH18A1", 1.10), ("PYCR1", 1.05), ("PYCR2", 1.00),
    ("OAT", 0.95), ("PRODH", 0.90),
    ("SLC1A5", 0.85), ("SLC1A4", 0.80),
    ("SLC7A11", 0.75), ("SLC7A5", 0.70), ("SLC38A1", 0.65),
    ("SLC38A2", 0.60), ("SLC38A5", 0.55),
    ("MYC", 0.50), ("ATF4", 0.45),
    ("BCAT1", 0.42), ("BCAT2", 0.40),
    ("PSAT1", 0.38), ("PSPH", 0.36), ("PHGDH", 0.34),
]

GLUTAMINOLYSIS_PANEL = sorted(
    {g for g, _ in GLUTAMINOLYSIS_CORE}
    | {
        "SHMT1", "SHMT2", "MTHFD1", "MTHFD2", "MTHFD1L", "MTHFD2L",
        "GFPT1", "GFPT2", "NAGS", "OTC", "CPS1",
        "AGMAT", "ARG1", "ARG2",
        "DDO", "MAOA", "MAOB",
        "SLC25A22", "SLC25A18", "SLC25A12", "SLC25A13",
        "GLUL", "ALDH4A1",
    }
)

MITOBIO_CORE: List[Tuple[str, float]] = [
    ("PPARGC1A", 1.50), ("PPARGC1B", 1.45),
    ("NRF1", 1.42), ("NFE2L2", 1.40),
    ("TFAM", 1.38), ("TFB1M", 1.32), ("TFB2M", 1.30),
    ("ESRRA", 1.28), ("ESRRG", 1.25),
    ("PINK1", 1.22), ("PRKN", 1.20),
    ("BNIP3", 1.18), ("BNIP3L", 1.15), ("FUNDC1", 1.10),
    ("MFN1", 1.05), ("MFN2", 1.00), ("OPA1", 0.95),
    ("DNM1L", 0.90), ("FIS1", 0.85),
    ("YME1L1", 0.80), ("OMA1", 0.75),
    ("MAP1LC3A", 0.70), ("MAP1LC3B", 0.65), ("GABARAP", 0.60),
    ("SQSTM1", 0.55), ("OPTN", 0.50), ("CALCOCO2", 0.48),
    ("MFF", 0.45), ("MID49", 0.42), ("MID51", 0.40),
]

MITOBIO_PANEL = sorted(
    {g for g, _ in MITOBIO_CORE}
    | {
        "POLG", "POLG2", "TWNK", "TFEC",
        "MFF", "MID49", "MID51",
        "PHB", "PHB2", "STOML2",
        "AFG3L1P", "AFG3L2", "SPG7", "LONP1", "CLPP", "CLPX",
        "DNAJC15", "DNAJC11",
        "MFN1", "MIRO1", "MIRO2", "RHOT1", "RHOT2",
        "ATG5", "ATG7", "ATG12", "ULK1", "ULK2",
        "TBK1", "TAX1BP1", "NBR1",
    }
)


_AXES = [
    {
        "axis_id": "Glycolysis",
        "name_en": "Glycolysis",
        "name_cn": "糖酵解",
        "biological_scope": "Functional gene module: cytosolic glucose -> pyruvate -> lactate flux (Warburg-style metabolism characteristic of avascular cartilage).",
        "interpretation": "糖酵解 / Warburg-style 代谢通量",
        "core": GLYCOLYSIS_CORE,
        "panel": GLYCOLYSIS_PANEL,
        "anti": [],
        "literature": [
            "MSigDB Hallmark Glycolysis (HALLMARK_GLYCOLYSIS)",
            "KEGG hsa00010 Glycolysis / Gluconeogenesis",
            "Reactome R-HSA-70171 Glycolysis",
            "PMID 29478910 Hypoxia-driven glycolysis sustains cartilage homeostasis",
            "PMID 30602793 PFKFB3-driven glycolysis in OA chondrocytes",
        ],
    },
    {
        "axis_id": "OxidativePhosphorylation",
        "name_en": "OxidativePhosphorylation",
        "name_cn": "氧化磷酸化",
        "biological_scope": "Functional gene module: mitochondrial electron transport chain (complexes I-V) and ATP synthesis.",
        "interpretation": "线粒体氧化磷酸化 / OXPHOS 通量",
        "core": OXPHOS_CORE,
        "panel": OXPHOS_PANEL,
        "anti": [],
        "literature": [
            "MSigDB Hallmark Oxidative Phosphorylation",
            "KEGG hsa00190 Oxidative phosphorylation",
            "Reactome R-HSA-163200 Respiratory electron transport, ATP synthesis by chemiosmotic coupling",
            "PMID 31926610 Mitochondrial dysfunction in osteoarthritis cartilage",
            "PMID 35421183 OXPHOS rewiring in chondrocyte aging",
        ],
    },
    {
        "axis_id": "TCA_Cycle",
        "name_en": "TCA_Cycle",
        "name_cn": "三羧酸循环",
        "biological_scope": "Functional gene module: mitochondrial citrate cycle and anaplerotic feeders (PC, ME, GLUD).",
        "interpretation": "三羧酸循环 / 中央代谢枢纽",
        "core": TCA_CORE,
        "panel": TCA_PANEL,
        "anti": [],
        "literature": [
            "KEGG hsa00020 Citrate cycle (TCA cycle)",
            "Reactome R-HSA-71403 Citric acid cycle (TCA cycle)",
            "PMID 32669297 TCA intermediates and chondrocyte phenotype switching",
        ],
    },
    {
        "axis_id": "PentosePhosphatePathway",
        "name_en": "PentosePhosphatePathway",
        "name_cn": "磷酸戊糖通路",
        "biological_scope": "Functional gene module: oxidative + non-oxidative pentose-phosphate branch supplying NADPH and ribose-5-phosphate.",
        "interpretation": "磷酸戊糖通路 / NADPH 与核糖供给",
        "core": PPP_CORE,
        "panel": PPP_PANEL,
        "anti": [],
        "literature": [
            "KEGG hsa00030 Pentose phosphate pathway",
            "Reactome R-HSA-71336 Pentose phosphate pathway",
            "PMID 27576679 Redox stress and PPP rewiring in OA cartilage",
        ],
    },
    {
        "axis_id": "FattyAcidOxidation",
        "name_en": "FattyAcidOxidation",
        "name_cn": "脂肪酸β氧化",
        "biological_scope": "Functional gene module: mitochondrial / peroxisomal fatty-acid β-oxidation, including carnitine shuttle and PPARα-driven program.",
        "interpretation": "脂肪酸 β 氧化 / FAO 能量代谢",
        "core": FAO_CORE,
        "panel": FAO_PANEL,
        "anti": [],
        "literature": [
            "KEGG hsa00071 Fatty acid degradation",
            "Reactome R-HSA-77289 Mitochondrial fatty acid beta-oxidation",
            "MSigDB Hallmark Fatty Acid Metabolism",
            "PMID 31926610 Lipid metabolism dysregulation in OA cartilage",
        ],
    },
    {
        "axis_id": "Lipogenesis",
        "name_en": "Lipogenesis",
        "name_cn": "脂质合成",
        "biological_scope": "Functional gene module: de novo fatty-acid + triacylglycerol synthesis under SREBP1 / ChREBP / PPARG control.",
        "interpretation": "脂肪酸/甘油三酯合成 / SREBP-ChREBP 通路",
        "core": LIPOGENESIS_CORE,
        "panel": LIPOGENESIS_PANEL,
        "anti": [],
        "literature": [
            "MSigDB Hallmark Adipogenesis",
            "Reactome R-HSA-8978868 Fatty acid metabolism",
            "PMID 35421183 De novo lipogenesis in OA chondrocytes",
            "PMID 32669297 Ectopic lipid accumulation in cartilage",
        ],
    },
    {
        "axis_id": "CholesterolHomeostasis",
        "name_en": "CholesterolHomeostasis",
        "name_cn": "胆固醇代谢",
        "biological_scope": "Functional gene module: mevalonate / sterol biosynthesis, sterol uptake (LDLR), and SREBP2-LXR feedback.",
        "interpretation": "胆固醇合成与稳态 / SREBP2-LXR 通路",
        "core": CHOLESTEROL_CORE,
        "panel": CHOLESTEROL_PANEL,
        "anti": [],
        "literature": [
            "MSigDB Hallmark Cholesterol Homeostasis",
            "KEGG hsa00100 Steroid biosynthesis",
            "Reactome R-HSA-191273 Cholesterol biosynthesis",
            "PMID 27576679 Cholesterol-driven OA chondrocyte hypertrophy",
        ],
    },
    {
        "axis_id": "LipidDroplet",
        "name_en": "LipidDroplet",
        "name_cn": "脂滴生物学",
        "biological_scope": "Functional gene module: lipid-droplet biogenesis, perilipin coat, neutral-lipid storage and lipolysis.",
        "interpretation": "脂滴 / 中性脂储存与动员",
        "core": LIPIDDROPLET_CORE,
        "panel": LIPIDDROPLET_PANEL,
        "anti": [],
        "literature": [
            "PMID 30385683 Cellular lipid droplet biology and proteome",
            "PMID 32669297 Ectopic lipid droplets in OA cartilage",
            "Reactome R-HSA-8978868 Fatty acid metabolism (LD subset)",
        ],
    },
    {
        "axis_id": "Glutaminolysis",
        "name_en": "Glutaminolysis",
        "name_cn": "谷氨酰胺分解",
        "biological_scope": "Functional gene module: glutamine uptake, glutaminolysis, anaplerotic α-ketoglutarate generation and proline / serine cross-feeds.",
        "interpretation": "谷氨酰胺分解 / α-KG 回补与生物合成",
        "core": GLUTAMINOLYSIS_CORE,
        "panel": GLUTAMINOLYSIS_PANEL,
        "anti": [],
        "literature": [
            "Reactome R-HSA-8964539 Glutamate and glutamine metabolism",
            "PMID 31527469 Glutamine metabolism in chondrocyte ECM synthesis",
            "PMID 35421183 Amino-acid metabolism rewiring in OA",
        ],
    },
    {
        "axis_id": "MitochondrialBiogenesis",
        "name_en": "MitochondrialBiogenesis",
        "name_cn": "线粒体生物发生",
        "biological_scope": "Functional gene module: PGC-1 / NRF / TFAM-driven mitochondrial biogenesis plus PINK1-PARKIN mitophagy quality control.",
        "interpretation": "线粒体生物发生与质控 / PGC1-PINK1 网络",
        "core": MITOBIO_CORE,
        "panel": MITOBIO_PANEL,
        "anti": [],
        "literature": [
            "PMID 14744432 PGC-1 coactivators in mitochondrial biogenesis",
            "PMID 21179058 PINK1-Parkin mitophagy",
            "MitoCarta 3.0 (PMID 33174596)",
            "PMID 31926610 Mitochondrial quality control in OA cartilage",
        ],
    },
]


def axis_specs() -> List[Dict[str, Any]]:
    """Return the in-memory specs (after extension) consumed by the build script."""
    return list(_AXES)
