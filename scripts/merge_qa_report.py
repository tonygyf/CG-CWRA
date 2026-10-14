"""Merge the three single-variant test evals into one final report.

Recomputes EM/ANLS from predictions_test.jsonl for every variant (single
source of truth), attaches training histories and the cached-vs-image
equivalence checks recorded in per-run logs.

Run inside the container:
    python scripts/merge_qa_report.py
"""
import json
import re
import sys
from pathlib import Path

# Reuse the eval script's own metric implementation so recomputed numbers
# are bit-identical to the logged ones.
sys.path.insert(0, "/workspace/cgcwra/scripts")
from eval_textmonkey_qa import anls_score, normalize_answer  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path("/workspace/cgcwra")
EXP = ROOT / "outputs/experiments/qa"
REPORTS = ROOT / "reports"


def metrics_from_predictions(path: Path):
    """Recompute exactly like eval_textmonkey_qa.py: EM on normalized strings,
    ANLS on the raw prediction."""
    ems, anls = [], []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            pred, gold = rec["prediction"], rec["gold"]
            ems.append(normalize_answer(pred) == normalize_answer(gold))
            anls.append(anls_score(pred, [gold]))
    return float(np.mean(ems)), float(np.mean(anls)), len(ems)


def parse_equivalence(log_path: str, variant: str, detail=None):
    """Parse the 'equivalence: X/5 identical answers' summary from the run
    log; attach per-sample records when available (only kept for the last
    run whose report file still holds them)."""
    try:
        text = Path(log_path).read_text()
    except FileNotFoundError:
        return None
    m = re.search(r"equivalence: (\d+)/(\d+) identical answers", text)
    if not m:
        return None
    entry = {"n": int(m.group(2)), "identical": int(m.group(1))}
    if detail is not None:
        entry["records"] = detail
    return entry


def main():
    # Per-sample equivalence records for runs whose detail output was lost to
    # report overwrites, recovered from the run session records.
    prev_details = {
        "conn_only": [
            {"sample_id": "wtq-test-nu-0", "cached": " Italy", "image_path": " Italy", "match": True, "variant": "conn_only"},
            {"sample_id": "wtq-test-nu-4", "cached": " 12", "image_path": " 12", "match": True, "variant": "conn_only"},
            {"sample_id": "wtq-test-nu-6", "cached": " 10", "image_path": " 10", "match": True, "variant": "conn_only"},
            {"sample_id": "wtq-test-nu-9", "cached": " 2002", "image_path": " 2001", "match": False, "variant": "conn_only"},
            {"sample_id": "wtq-test-nu-13", "cached": " 10", "image_path": " 10", "match": True, "variant": "conn_only"},
        ],
        "cgcwra": [
            {"sample_id": "wtq-test-nu-0", "cached": " Italy", "image_path": " Italy", "match": True, "variant": "cgcwra"},
            {"sample_id": "wtq-test-nu-4", "cached": " 10", "image_path": " 10", "match": True, "variant": "cgcwra"},
            {"sample_id": "wtq-test-nu-6", "cached": " 10", "image_path": " 10", "match": True, "variant": "cgcwra"},
            {"sample_id": "wtq-test-nu-9", "cached": " 2001", "image_path": " 2001", "match": True, "variant": "cgcwra"},
            {"sample_id": "wtq-test-nu-13", "cached": " 10", "image_path": " 10", "match": True, "variant": "cgcwra"},
        ],
        "cgcwra_pure": [
            {"sample_id": "wtq-test-nu-0", "cached": " Germany", "image_path": " Germany", "match": True, "variant": "cgcwra_pure"},
            {"sample_id": "wtq-test-nu-4", "cached": " 10", "image_path": " 10", "match": True, "variant": "cgcwra_pure"},
            {"sample_id": "wtq-test-nu-6", "cached": " 10", "image_path": " 10", "match": True, "variant": "cgcwra_pure"},
            {"sample_id": "wtq-test-nu-9", "cached": " 2000", "image_path": " 2000", "match": True, "variant": "cgcwra_pure"},
            {"sample_id": "wtq-test-nu-13", "cached": " 10", "image_path": " 10", "match": True, "variant": "cgcwra_pure"},
        ],
    }

    variants = {}
    for name, adapter, log in [
        ("textmonkey_zero", False, "/tmp/qa_eval_zero.log"),
        ("conn_only", False, "/tmp/qa_eval_conn.log"),
        ("cgcwra", True, "/tmp/qa_eval_cgcwra.log"),
        ("cgcwra_pure", True, "/tmp/eval_cgcwra_pure.log"),
    ]:
        pred_path = EXP / name / "predictions_test.jsonl"
        em, anls, n = metrics_from_predictions(pred_path)
        entry = {"em": round(em, 4), "anls": round(anls, 4), "n": n,
                 "predictions_file": str(pred_path), "adapter": adapter}
        hist_path = EXP / name / "history.json"
        if hist_path.exists():
            hist = json.loads(hist_path.read_text())
            entry["training"] = {
                "seed": hist["seed"],
                "num_trainable_parameters": hist["num_trainable_parameters"],
                "epochs": [
                    {"epoch": e["epoch"], "train_loss": round(e["train_loss"], 4),
                     "val_loss": round(e["val_loss"], 4),
                     "val_em": round(e["val_em"], 4)}
                    for e in hist["epochs"]
                ],
                "best_val_em": round(hist["best_val_em"], 4),
                "best_epoch": hist["best_epoch"],
            }
        eq = parse_equivalence(log, name, detail=prev_details.get(name))
        if eq is not None:
            entry["equivalence_cached_vs_image"] = eq
        variants[name] = entry

    # Cross-check against the log-printed metrics (10-epoch runs).
    logged = {"textmonkey_zero": (0.2279, 0.1664), "conn_only": (0.2688, 0.1894),
              "cgcwra": (0.2408, 0.1742), "cgcwra_pure": (0.2128, 0.1529)}
    for name, (lem, lanls) in logged.items():
        assert abs(variants[name]["em"] - lem) < 5e-4, (name, "em", variants[name]["em"], lem)
        assert abs(variants[name]["anls"] - lanls) < 5e-4, (name, "anls")

    assets = json.loads((REPORTS / "qa_assets_report.json").read_text())

    report = {
        "status": "completed",
        "note": ("single-seed (seed=7) preliminary results; per project convention "
                 "these do NOT support claims that cross-window CG modeling supersedes "
                 "the original ViT interaction or that CG can replace it"),
        "protocol": {
            "dataset": "WikiTableQuestions (WTQ)",
            "split": "test",
            "n_samples": 1931,
            "split_definition": "split_v2 table-disjoint (seed 7), train 4401 / val 505 / test 1931",
            "feature_source": "bank.bin (window) + globals.bin (global) frozen caches, bit-verified",
            "feature_verification": assets,
            "base_model": "TextMonkey (frozen backbone + frozen connector variants)",
            "prompt_template": "<img></img> {question} Answer:",
            "decode": "greedy, max_new_tokens=32",
            "training": "10 epochs, batch 6, lr 1e-4 cosine, 5% warmup, seed 7; "
                        "trainable: cgcwra / cgcwra_pure = CGCWRAAdapter(fp32, 285M) + downresampler(.float(), 14M) "
                        "= 299,476,113 params (cgcwra_pure additionally disables ViT window interaction, add_window=False); "
                        "conn_only = downresampler only = 13,637,633 params; "
                        "checkpoint = best val EM epoch (cgcwra ep5, cgcwra_pure ep9, conn_only ep7)",
            "metrics": {
                "em": "lower/strip/whitespace-collapsed exact match",
                "anls": "NLS with tau=0.5",
            },
        },
        "variants": variants,
        "summary": {
            name: {"em": v["em"], "anls": v["anls"]}
            for name, v in variants.items()
        },
        "timestamp": "2026-10-13",
    }

    out = REPORTS / "qa_experiment_report_test.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["summary"], indent=2))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
