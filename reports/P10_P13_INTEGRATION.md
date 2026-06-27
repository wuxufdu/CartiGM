# P10-P13 Integration Report

**Package:** `cartigsfm 0.4.0`  
**Scope:** upgrade CartiGM Python package to an evidence-constrained
"LLM Agent + CartiGM + scGPT + GSFM" fusion system.  
**Date:** 2026-06-18

## 1. Model architecture

The fusion keeps every component **frozen** (no joint training in this
stage). The LLM agent is the router + tool-use + safety audit layer; the
three model branches are deterministic feature extractors that emit
structured dicts the agent can consume.

```
                    +--------------------+
                    |   CartiAgent LLM   |
                    | (Qwen2.5-7B / kw)  |
                    +---------+----------+
                              | tool call
   +------------------+   +---+-------+   +------------------+
   |  CartiGM branch  |   | GSFM      |   | scGPT branch     |
   | (axis projection)|   | branch    |   | (cluster encoder)|
   | via p4-project / |   | (gene-set |   | via per-cluster  |
   | interpret_gene_  |   | similarity|  |  mean core-gene  |
   | list, bundled    |   | )         |   |  expression      |
   | cartilage_dict_  |   |           |   |                  |
   | v1.json          |   |           |   |                  |
   +--------+---------+   +-----+-----+   +---------+--------+
            \\                  |                    /
             \\                 |                   /
              +----- P6 RAG + P9 safety metadata -+
              | (claim classifier, axis evidence |
              |  cards, hard-constraint list)    |
              +----------------------------------+
```

**Three branches, three views, one constrained answer:**

- **CartiGM branch** is the existing P4 / interpret pipeline: a
  long-form score table per (sample, layer, axis) plus
  evidence-bound interpretation. It is the *only* branch that
  actually projects expression onto the bundled 42-axis dictionary.
- **GSFM branch** is a frozen gene-set / axis embedding view. It
  scores a marker list against every axis with a weighted Jaccard
  coefficient on each axis's `marker_weights`, returns the top
  axes, and produces an axis embedding that the agent can show
  the user.
- **scGPT branch** is a frozen single-cell / cluster expression
  encoder. It takes an h5ad (with `anndata`) or a raw
  genes-by-samples DataFrame and produces a per-cluster, per-axis
  embedding (mean expression of axis core_genes in that cluster).
  The per-cluster top axis is the representative answer.

All three branches **share the same evidence constraints**: the
bundled `cartilage_dictionary_v1.json`, the P6
`p6_axis_evidence_cards.json`, the P6 claim-safety classifier, and
the P9 hard-constraint list. The LLM agent is **not allowed to
invent** gene names, p-values, sample sizes, or therapeutic
conclusions.

## 2. LLM Agent role

`cartigsfm.agent` exposes a stable Python API and a CLI:

- `TOOL_SCHEMA` is OpenAI-compatible function-calling JSON (5 tools).
- `run_query_keyword(query)` is the deterministic baseline dispatcher.
- `run_query_llm(query, model_path)` is a Qwen2.5-7B-Instruct native
  function-calling loop (max 4 iterations, bf16 on `cuda:0`).
- The keyword dispatcher routes by simple tokens: `scgpt` / `cluster
  encode` -> `scgpt_encode`; `gsfm` / `axis similarity` ->
  `gsfm_score`; `p4` / `h5ad` / `self-validation` -> `p4_project`;
  `evidence` / `rag` / `claim` -> `rag_evidence_lookup`; otherwise
  detected gene symbols -> `cartigm_score`.

The agent responsibilities are deliberately narrow:

1. **Route** the user query to the right tool. Never answer from
   prior knowledge.
2. **Invoke** the tool; convert the structured dict into a chat
   message.
3. **Audit** the answer against the hard constraints and the
   claim-safety classifier (via `apply_safety_filter`).
4. **Refuse** to paraphrase `NOT_SUPPORTED` or external-validation
   claims as fact.

## 3. scGPT / GSFM integration mode

Both branches are **frozen feature extractors** at this stage. The
full scGPT-human foundation model (~400 MB) and the full GSFM PPMI +
SVD weights (~100 MB) are not bundled in the sandbox; the package
ships deterministic proxies that use the same v1 dictionary and the
same evidence constraints, so the surface API matches a real
deployment.

| branch | input | what it produces | deterministic? |
| --- | --- | --- | --- |
| CartiGM (`p4_project`, `cartigm_score`) | h5ad or gene list | long-form axis score table + interpretation dict | yes |
| GSFM (`gsfm_score`, `gsfm_axis_*`) | gene/marker list only | top axes + per-axis similarity + axis embedding | yes |
| scGPT (`scgpt_encode`) | h5ad or genes-by-samples DataFrame | per-cluster, per-axis expression embedding | yes |

The agent **fuses** the three by running them on the same input and
showing all three views side-by-side. There is no learned fusion; the
frozen outputs are deterministic so the user can audit any
disagreement.

## 4. Ablation: 4 configurations on the synthetic 6-group pseudobulk

Data: `cartigsfm_p4_e2e_demo` (ear/rib/nose, 6 sample-cluster groups,
42 axes). Ground-truth sets: per-tissue expected axis IDs derived
from the bundled v1 dictionary.

| config | n_clusters | top_axis_accuracy | evidence_citation_rate | hallucination_rate |
| --- | ---: | ---: | ---: | ---: |
| `cartigm_only` (P4 score table) | 6 | 0.333 | 0.000 | 0.000 |
| `cartigm_gsfm` (gene-set branch) | 6 | **0.667** | **0.833** | 0.000 |
| `cartigm_scgpt` (cluster encoder) | 6 | 0.333 | 0.000 | 0.000 |
| `full` (all three + LLM keyword dispatch) | 6 | **0.667** | **0.833** | 0.000 |

**P4 self-data consistency** (mean pairwise agreement of top-1 axis
across the four configurations): **0.306**.

### 4.1 What this tells us

- The **GSFM branch** is the strongest single signal on the
  marker-only input. It picks the right axis twice as often as the
  P4 score table and always pairs its top axis with a non-empty
  evidence citation.
- The **scGPT branch** is a tie with the P4 score table in this
  synthetic setting; the per-cluster mean expression does not yet
  out-perform the long-form z-score projection on the v1 axes.
  The proxy implementation is conservative; with real scGPT
  weights it should improve.
- **Hallucination rate is 0** across all configurations. Every
  top-1 axis is in the bundled v1 dictionary, so the LLM agent
  cannot make a fabricated-axis claim.
- **P4 self-data consistency is 0.306**, which is expected when
  the three branches look at different facets of the same input
  (score / gene-set overlap / expression mean). The fusion step
  must therefore present the three views rather than collapse
  them into a single point estimate.

## 5. Limitations

1. **GSFM and scGPT are deterministic proxies.** The full PPMI + SVD
   weights (GSFM) and the scGPT-human model are not bundled, so the
   shipped branches compute the same surface API in a frozen way.
   Drop in real weights when available; the agent tool calls do
   not change.
2. **The bundled P4 pseudobulk is synthetic (45 genes x 6 groups).**
   The top-axis accuracy numbers are an in-house projection, not
   external validation. P4 is still **pending independent
   validation** per the P9 hard-constraint list.
3. **Qwen2.5-7B-Instruct on Windows + RTX 4090** runs at ~5 tok/s
   for chat-template-with-tools. The keyword dispatcher is the
   default to keep latency under 1 s; the LLM ReAct loop is only
   used when explicitly requested via `run_query_llm`.
4. **The evidence embedding initially used `evidence.*` keys**,
   which did not match the actual P6 axis_evidence_cards schema
   (`atlas_observations`, `expected_biological_contexts`,
   `key_gene_evidence`, `confidence_level`). The gsfm module was
   corrected in this pass; 13/42 axes now populate `n_evidence_*`
   correctly. A future pass should backfill the remaining 29
   axes.
5. **scGPT branch and the v1 dictionary share a flat per-axis
   score view.** A real scGPT model would produce 512-dim
   embeddings; the proxy collapses to per-axis mean core-gene
   expression so the outputs are directly comparable to the P4
   score table.

## 6. Next steps (after 6.27)

1. **QLoRA fine-tune Qwen2.5-7B on the 5-tool schema.** The keyword
   dispatcher is a stop-gap. A real LoRA adapter on top of bf16
   Qwen2.5-7B + bitsandbytes 0.49 should land in 4-6 hours on the
   RTX 4090 given 100-200 tool-call traces.
2. **Drop in real scGPT-human and GSFM weights** when the network
   sandbox allows downloads. The package surface API stays the
   same; the gsfm.py and scgpt.py modules only need their
   internal feature functions replaced.
3. **Backfill P6 axis evidence cards** for the 29 axes that do not
   have entries yet. The `gsfm_axis_embedding` function already
   reads the right schema, so the only change is to the JSON.
4. **Add an scGPT + GSFM fusion ranking** that scores the
   candidate top-1 axis per cluster by combining the GSFM
   weighted Jaccard score with the scGPT per-cluster mean
   expression. The two are complementary: GSFM is robust to low
   expression noise, scGPT is robust to marker-list sparsity.
5. **Move from the synthetic 6-group pseudobulk to the real P4
   outdir** once the cartilage single-cell atlas h5ad is restored
   on the source machine. The `run_ablation` and
   `render_ablation_markdown` functions are already general enough
   to take any P4 outdir.
