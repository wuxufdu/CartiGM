# CartiGSFM — Related Work (literature scan, June 2026)

Searched 2024-01 to 2026-06 via Crossref for: gene-set foundation models,
single-cell foundation models, cell-type-specific gene-set analysis, and
cartilage / OA single-cell atlases. The list below is curated to the work
most relevant for designing a cartilage gene-set foundation model.

## 1. The reference work — GSFM

- **GSFM: A gene set foundation model pre-trained on a massive collection of
  diverse gene sets** — Patterns 2026,
  [10.1016/j.patter.2026.101565](https://doi.org/10.1016/j.patter.2026.101565)
  (preprint: bioRxiv 2025-05-30, [10.1101/2025.05.30.657124](https://doi.org/10.1101/2025.05.30.657124)).
  This is the model the user pointed at. Pre-trains on a large pool of
  diverse gene sets and learns set-level representations. CartiGSFM is the
  cartilage-specialised analogue: same paradigm, narrower domain, with
  mechanism-aware specificity instead of pure name retrieval.

## 2. Other gene-set / pathway language models (2024-2025)

- **GeneInsight: Condensing Gene Set Knowledge via Language Models** —
  bioRxiv 2025, [10.1101/2025.07.07.663611](https://doi.org/10.1101/2025.07.07.663611).
  Uses LLMs to compress and represent gene-set knowledge. Useful prior art
  for a "summarise a cartilage gene set" task.
- **Gene-R1: reasoning with data-augmented lightweight LLMs for gene set
  analysis** — F1000Research 2025,
  [10.7490/f1000research.1120383.1](https://doi.org/10.7490/f1000research.1120383.1).
  Reasoning-style fine-tune; relevant if we later let the model justify
  why a query maps to a subtype.
- **Uncovering Latent Biological Function Associations through Gene Set
  Embeddings** — bioRxiv 2024,
  [10.1101/2024.10.10.617577](https://doi.org/10.1101/2024.10.10.617577).
  Embedding-space approach to gene-set similarity; this is the closest
  baseline for the v0.4 specificity step we already built.
- **Leveraging cell type-specificity for gene set analysis of single cell
  transcriptomics** — bioRxiv 2024,
  [10.1101/2024.09.25.615040](https://doi.org/10.1101/2024.09.25.615040).
  Justifies the need for the cell-type axis we encode in
  `subtype_dictionary` / specificity scoring.
- **Benchmarking Cell Type and Gene Set Annotation by Large Language Models
  with AnnDictionary** — bioRxiv 2024,
  [10.1101/2024.10.10.617605](https://doi.org/10.1101/2024.10.10.617605).
  Benchmark we should evaluate CartiGSFM against once a model layer
  exists.
- **Large Language Model Consensus Substantially Improves the Cell Type
  Annotation Accuracy for scRNA-seq Data** — bioRxiv 2025,
  [10.1101/2025.04.10.647852](https://doi.org/10.1101/2025.04.10.647852).
  Multi-LLM consensus ideas; relevant for deduplicating noisy literature
  evidence (cf. our v0.5 anchor-rebuild step).

## 3. Single-cell foundation models — context for the modelling layer

- **A cross-species foundation model for single cells** — Cell Research
  2024, [10.1038/s41422-024-01045-9](https://doi.org/10.1038/s41422-024-01045-9).
  Closest production-grade scFoundation; user requested human-only so
  this becomes a candidate for human-pretrain weights to fine-tune from.
- **scGPT-spatial: Continual Pretraining of Single-Cell Foundation Model
  for Spatial Transcriptomics** — bioRxiv 2025,
  [10.1101/2025.02.05.636714](https://doi.org/10.1101/2025.02.05.636714).
  If we later want spatial cartilage data (zonal context).
- **CLM-X: A multimodal single-cell foundation model with flexible
  multi-way Transformer for unified scRNA-seq and scATAC-seq** — preprint
  2026, [10.64898/2026.02.17.704943](https://doi.org/10.64898/2026.02.17.704943).
  scRNA + scATAC joint, relevant if user's local data adds ATAC.
- **Mouse-Geneformer: cross-species utility** — bioRxiv 2024,
  [10.1101/2024.09.09.611960](https://doi.org/10.1101/2024.09.09.611960).
  Recipe for single-species pretrains; supports our human-only choice.
- **Sparse Autoencoders Reveal Interpretable Cell-Type Programs in
  Single-Cell Foundation Model Representations** — SSRN 2026,
  [10.2139/ssrn.6304512](https://doi.org/10.2139/ssrn.6304512).
  Interpretability layer that pairs naturally with our consensus +
  specificity dictionaries.
- **Two Axes in the Gene-Embedding Space of Single-Cell Foundation
  Models** — SSRN 2026,
  [10.2139/ssrn.6577900](https://doi.org/10.2139/ssrn.6577900).
  Suggests a co-functional axis vs. a co-expression axis; CartiGSFM's
  v0.4 specificity already separates the "cartilage shared core" axis
  from "subtype identity" axis.
- **Assessing Scale and Predictive Diversity in Models for Single-Cell
  Transcriptomics based on Geneformer** — bioRxiv 2025,
  [10.1101/2025.11.04.686458](https://doi.org/10.1101/2025.11.04.686458).
  Scale-vs-utility analysis; argues against blindly scaling tokens, in
  favour of curated training sets — exactly the bet we are making.

## 4. Cartilage / OA single-cell atlases (2024-2026) — possible v0.7 seeds

- **Standardizing single-cell approaches to osteoarthritis: Toward a
  comprehensive cellular atlas** — Osteoarthritis Cartilage 2026,
  [10.1016/j.joca.2025.08.016](https://doi.org/10.1016/j.joca.2025.08.016).
  Nomenclature standard; lets us reconcile our 19 subtypes with field
  consensus before we lock the dictionary.
- **Subcellular spatial and single-cell transcriptional atlas of TMJ-OA**
  — Osteoarthritis Cartilage 2025,
  [10.1016/j.joca.2025.02.565](https://doi.org/10.1016/j.joca.2025.02.565).
  Adds TMJ-OA chondrocyte subtypes we don't yet cover.
- **Single-cell resolution mapping of OA-associated chondrocyte dynamics
  using high-density CMOS MEA platforms** — Osteoarthritis Cartilage 2026,
  [10.1016/j.joca.2026.01.561](https://doi.org/10.1016/j.joca.2026.01.561).
  Functional state markers worth pulling.
- **Single-cell transcriptome-wide MR + colocalisation for regulatory
  T-cell-specific OA loci** — Osteoarthritis Cartilage 2026,
  [10.1016/j.joca.2026.01.454](https://doi.org/10.1016/j.joca.2026.01.454).
  Adds a regulatory immune-cell axis adjacent to our subtype model.
- **Cross-species scRNA-seq of adipose-derived MSCs in OA** —
  Osteoarthritis Cartilage 2025,
  [10.1016/j.joca.2025.02.056](https://doi.org/10.1016/j.joca.2025.02.056).
  AD-MSC markers; refines our `MSC_Progenitor` axis.
- **Single-cell atlas reveals age-related cellular shifts underlying
  fibrosis in murine synovium** — Osteoarthritis Cartilage 2026,
  [10.1016/j.joca.2026.01.094](https://doi.org/10.1016/j.joca.2026.01.094).
  Synovial-fibroblast aging signatures; complements our v0.5
  SynovialFibroblast category.
- **scRNA-seq of mesenchymal populations from murine knees: pathways
  altered in age-associated OA** — Osteoarthritis Cartilage 2024,
  [10.1016/j.joca.2024.03.088](https://doi.org/10.1016/j.joca.2024.03.088).
  Aging-vs-OA stratification.
- **A Single Instance of Joint Overloading Results in Persistent Changes
  to the Synovial Cell Landscape** — Osteoarthritis Cartilage 2026,
  [10.1016/j.joca.2026.01.072](https://doi.org/10.1016/j.joca.2026.01.072).
  Mechano-loading state markers — feeds directly into our v0.6
  Mechanotransduction category.

## 5. How CartiGSFM positions against this literature

| Axis | Field state (2024-2026) | CartiGSFM v0.6 |
|---|---|---|
| Pretraining corpus | broad gene sets (GSFM) | cartilage-narrow, 13,227 sets, paper-anchored |
| Cell-type axis | learned from scRNA atlases | curated dictionary + signature expansion |
| Specificity | implicit in embedding space | explicit TF-IDF score, audit-traceable |
| Provenance | usually opaque | per-category set_id + DOI evidence |
| Domain knowledge | absent or minimal | 15 paper-attributed seeds (Ji 2019, Decker 2017, Mizuhashi 2018, Tam 2020, Sun 2020) |
| Scope | whole organism | human-only by user choice; ortholog mapping deferred |

## 6. Open questions to revisit before model training

1. Which scRNA atlases will the user merge in? Each adds a fresh
   subtype-marker layer that should re-seed v0.7 (especially the 2026
   TMJ-OA, MEA OA, joint-overload synovium papers above).
2. Embedding architecture: do we fine-tune scFoundation/Geneformer on
   the cartilage corpus, or train a smaller cartilage-specific
   transformer from scratch on the 45-category dictionary as supervision?
3. Evaluation: AnnDictionary benchmark (#2 above) + held-out canonical
   queries (already 13/13 pass on dictionary level) + scRNA atlas
   cell-type recovery on the user's local integration.
