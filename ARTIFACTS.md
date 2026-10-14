# CG-CWRA — Artifact & Storage Map

Snapshot: 2026-10-14. Host repo root: `/home/gyf/CG-CWRA`
(container mount: `/workspace/cgcwra`).

This file is the single index of (a) where the **verification artifacts** live
and (b) which files are **meaningful** and therefore intended for the
HuggingFace upload.

---

## 1. Storage layout (high level)

| Location | Contents | Notes |
|---|---|---|
| `checkpoints/qa/` | Archived best checkpoints (one per variant) + training histories | git-ignored, HF upload target |
| `outputs/experiments/qa/<variant>/` | Per-run `checkpoint_best.pt`, `history.json`, `predictions_test.jsonl` | git-ignored |
| `reports/` | Small JSON verification/result reports | **versioned in git** |
| `data/processed/wtq_core/` | WTQ core dataset: `bank/` (20G), `images/` (129M), `split_v2/` (44M), `train|val|test.jsonl` | git-ignored |
| `artifacts/pretrained/` | Base weights: `TextMonkey` (19G), `Qwen-VL` (19G) | git-ignored |
| `scripts/` | Data build, training, evaluation, merge, container runner | versioned |
| `src/cgcwra/` | Library code incl. `models/textmonkey*.py`, `transformers_compat.py` | versioned |

---

## 2. Verification artifacts (where they live)

| Path | Produced by | What it verifies |
|---|---|---|
| `reports/qa_assets_report.json` | `scripts/prep_textmonkey_qa_assets.py` | Frozen window/global feature caches (`bank.bin`/`globals.bin`) match image-path recompute (bit-level check) |
| `reports/qa_experiment_report_test.json` | `scripts/merge_qa_report.py` | Merged test EM/ANLS per variant + `equivalence_check` (cached vs image-path decoding) |
| `reports/wtq_core_data_report.json` | `scripts/generate_wtq_core_data.py` | WTQ core build: 4-window config, table-disjoint split stats |
| `reports/wtq_nointer_data_report.json` | `scripts/generate_wtq_core_data.py --disable-vit-window-attention` | nointer feature bank (`add_window=False`) |
| `outputs/experiments/qa/<variant>/predictions_test.jsonl` | `scripts/eval_textmonkey_qa.py` | Per-sample test predictions — **source of truth** for EM/ANLS recomputation |
| `checkpoints/qa/<variant>_history.json` | `scripts/train_textmonkey_qa.py` | 10-epoch train/val curves (overfitting onset evidence) |
| `scripts/verify_ablation_consistency.py` | — | Ablation-variant config consistency (window interaction on/off) |
| `.trae/论文修改对齐.md` | audit session | Paper vs code vs TextMonkey image pipeline alignment audit |

Recompute from scratch (source of truth = predictions, not logs):

```
python scripts/merge_qa_report.py     # inside container
```

---

## 3. Meaningful files → HuggingFace upload

Variant legend: `inter` = TextMonkey ViT window interaction; `CG` = CG-CWRA adapter.

| File | Size | Contents | Variant |
|---|---|---|---|
| `checkpoints/qa/cgcwra_best.pt` | 1.2G | `adapter` (285M) + `downresampler`; best val-EM epoch 5 | inter ON + CG |
| `checkpoints/qa/cgcwra_pure_best.pt` | 1.2G | same module; ViT window interaction OFF; epoch 9 | inter OFF + CG |
| `checkpoints/qa/conn_only_best.pt` | 54M | `downresampler` only (no adapter); epoch 7; val-EM 0.2851 | inter ON, no CG |
| `checkpoints/qa/conn_only_nointer_best.pt` | 54M | `downresampler` only; window interaction OFF; epoch 7 | inter OFF, no CG |
| `checkpoints/qa/*_history.json` | ~2.4K each | 10-epoch curves (seed 7) | all |
| `reports/qa_experiment_report_test.json` | 7.9K | Protocol + per-variant EM/ANLS + equivalence checks | all |

Per-sample predictions (optional, ~290K each):
`outputs/experiments/qa/{cgcwra,cgcwra_pure,conn_only,textmonkey_zero}/predictions_test.jsonl`

Suggested HF model-repo layout:

```
cg-cwra-wtq/
├── cgcwra_best.pt
├── cgcwra_pure_best.pt
├── conn_only_best.pt
├── conn_only_nointer_best.pt
├── README.md            # copy sections 3-4 here
└── histories/{cgcwra,cgcwra_pure,conn_only,conn_only_nointer}_history.json
```

---

## 4. Protocol at a glance

* Dataset: WikiTableQuestions (WTQ); table-disjoint `split_v2` (seed 7), train 4401 / val 505 / test 1931.
* Base: TextMonkey frozen (backbone + connector untouched except trained variants); image path = 896×896 windows (4) + 448 global, 768 visual tokens.
* Training: 10 epochs, batch 6, lr 1e-4 cosine (5% warmup), seed 7; checkpoint = best val-EM epoch.
* Metrics: EM = normalized exact match; ANLS = normalized Levenshtein similarity (τ=0.5). Decoding: greedy, `max_new_tokens=32`.
* Single seed — differences below ~0.02 EM are within run-to-run noise; **no** claim that CG supersedes / replaces TextMonkey's ViT interaction is supported yet.

---

## 5. Pending before final submission

* `conn_only_nointer` test evaluation → complete the 2×2 matrix (currently 4/5).
* Paper wording: "patch 特征" (5 occurrences in `bare_jrnl.tex`) vs implementation operating on per-window attn_pool tokens — see `.trae/论文修改对齐.md` issue #1.
