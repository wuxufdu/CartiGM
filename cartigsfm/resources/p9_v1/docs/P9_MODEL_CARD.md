# CartiGSFM-LLM P9 Model Card

## Status
**Actually trained.** P9 is a small LoRA adapter on top of Qwen2.5-0.5B-Instruct,
fine-tuned on the P8 evidence-grounded cartilage instruction dataset.
P9 is **not** a domain foundation LLM and **not** externally validated.

## Base Model
- `Qwen/Qwen2.5-0.5B-Instruct` (494.03M params)
- Loaded in fp32 on Apple Silicon MPS (16 GB unified memory)
- `bitsandbytes` / QLoRA disabled (no CUDA on MPS)

## Adapter
| field | value |
|---|---|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Trainable params | 2.16M (0.436% of base) |
| Adapter file | `adapter/adapter_model.safetensors` (8.7 MB) |

## Training Run
| field | value |
|---|---|
| Train samples used | 60 (of 361 available) |
| Valid samples used | 10 (of 75 available) |
| Max sequence length | 320 tokens |
| Effective batch size | 2 (bs=1, accum=2) |
| Epochs | 1 |
| Learning rate | 2e-4 (cosine, warmup 0.03) |
| Steps | 30 |
| Wall time | 16.9 min |
| Train loss start → end | 2.859 → 1.860 |
| Train loss average | 2.259 |
| Eval loss final | 1.578 |

The 60-sample / 1-epoch / max_len=320 configuration was chosen because the
full 361-sample / max_len=512 nohup run was killed by macOS due to MPS
unified-memory pressure.

## Intended Use
- Evidence-grounded interpretation of CartiGSFM atlas-internal results.
- Reviewer-safe rewriting of CartiGSFM-related claims.
- Refusal of unsupported claims (external validation, trained LLM,
  therapeutic targeting, Microtia identity loss, OA inflammation
  significance, causal mechanism).
- Conservative handling of P4 / in-house single-cell context.

## Out-of-Scope Use
- Clinical decision support.
- Multi-center / external cohort validation claims.
- Mechanistic / causal claims based on CartiGSFM scores.
- Any extension beyond cartilage-domain CartiGSFM evidence registry.

## Evaluation Snapshot (n = 17, P4-augmented subset)
| system | refusal | hallucination | evidence cite | ext-val overclaim | P4 acc |
|---|---|---|---|---|---|
| base | 0.18 | 0.12 | 0.35 | 0.12 | 0.00 |
| rag_only | 0.55 | 0.00 | 0.71 | 0.00 | 0.00 |
| lora_only | **0.73** | 0.06 | **0.94** | 0.06 | **0.80** |
| lora_rag | 0.64 | 0.06 | 0.76 | 0.06 | 0.80 |

LoRA training improved unsupported-claim refusal by +55 percentage points
over base, evidence citation by +59pp, P4 case handling by +80pp, and
reduced external-validation overclaim by 6pp.

## Known Limitations
- Trained on a small (60-sample) subset; generalization is limited.
- Single base model; no comparison vs Qwen2.5-3B/7B or Llama-3.x.
- Clinical-validation overreach prompt still leaks through `lora_only`
  and `lora_rag` once each (`eval_p4_clinical_overreach`); needs more
  refusal samples or output guardrail.
- Adapter is FP32 / MPS; not yet packed for cross-platform inference.

## Reproducibility
```
HF_ENDPOINT=https://hf-mirror.com PYTORCH_ENABLE_MPS_FALLBACK=1 \
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
python3 scripts/train_p9_lora.py \
    --epochs 1 --max-len 320 --accum 2 --bs 1 --lr 2e-4 \
    --max-train 60 --max-valid 10 \
    --output adapter --log logs/p9_training_log.txt
```

Evaluation:
```
HF_ENDPOINT=https://hf-mirror.com PYTORCH_ENABLE_MPS_FALLBACK=1 \
python3 scripts/p9_evaluate.py \
    --eval-jsonl jsonl/p9_p4_augmented_eval.jsonl \
    --max-new-tokens 128
```
