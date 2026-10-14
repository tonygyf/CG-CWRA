#!/usr/bin/env python
"""消融一致性核查：数据划分、库行序、以及 add_window 开关的因果有效性。"""

import json
import sys
from pathlib import Path

ROOT = Path("/workspace/cgcwra")
ok = True


def table_ids(p):
    ids = {"train": set(), "val": set()}
    for line in Path(p).open():
        r = json.loads(line)
        ids[r["split"]].add(r["metadata"]["table_id"])
    return ids


a = table_ids(ROOT / "data/processed/wtq_core/split_v2/train.jsonl")
b = table_ids(ROOT / "data/processed/wtq_core_nointer/split_v2/train.jsonl")
same_train = a["train"] == b["train"]
same_val = a["val"] == b["val"]
ok &= same_train and same_val
print(f"train_tables_equal={same_train} ({len(a['train'])}/{len(b['train'])})")
print(f"val_tables_equal={same_val} ({len(a['val'])}/{len(b['val'])})")

ta = [json.loads(l)["metadata"]["table_id"] for l in open(ROOT / "data/processed/wtq_core/split_v2/test.jsonl")]
tb = [json.loads(l)["metadata"]["table_id"] for l in open(ROOT / "data/processed/wtq_core_nointer/split_v2/test.jsonl")]
same_test = ta == tb
ok &= same_test
print(f"test_identical={same_test} ({len(ta)})")

ma = json.load(open(ROOT / "data/processed/wtq_core/bank/bank.json"))
mb = json.load(open(ROOT / "data/processed/wtq_core_nointer/bank/bank.json"))
keys_a = [k for k in ma if isinstance(ma[k], (list, dict)) and k != "shape"]
print("bank.json keys:", list(ma.keys()), "|", list(mb.keys()))
row_a = json.dumps(ma, sort_keys=True)
row_b = json.dumps(mb, sort_keys=True)
same_bank_meta = row_a == row_b
print(f"bank_meta_equal={same_bank_meta}")
ok &= same_bank_meta

# 因果验证：同一张表图，add_window True vs False 的特征必须不同
import torch  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from cgcwra.models.textmonkey import load_bundle  # noqa: E402
from cgcwra.models.textmonkey_inference import encode_visual_tokens  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

bundle = load_bundle(ROOT / "artifacts/pretrained/TextMonkey", dtype=torch.bfloat16, device="cuda")
model = bundle.model
img = Image.new("RGB", (896, 896), "white")
d = ImageDraw.Draw(img)
d.rectangle((40, 40, 856, 200), fill=(30, 90, 160))
d.text((80, 90), "Causal Check Table", fill="white")
d.text((80, 300), "col_a  col_b  col_c", fill="black")
path = ROOT / "outputs/smoke/causal_check.png"
img.save(path)

wb = model.transformer.visual.transformer
f_on = encode_visual_tokens(model, [str(path)])
wb.add_window = False
f_off = encode_visual_tokens(model, [str(path)])
wb.add_window = True
delta = (f_on.tokens.float() - f_off.tokens.float()).abs()
print(f"window_attn causal check: shape={tuple(f_on.tokens.shape)} "
      f"mean_abs_delta={delta.mean():.6f} changed_fraction={(delta > 1e-4).float().mean():.4f}")
ok &= delta.mean() > 1e-3
print("ALL_CHECKS_PASSED" if ok else "CHECKS_FAILED")
sys.exit(0 if ok else 1)
