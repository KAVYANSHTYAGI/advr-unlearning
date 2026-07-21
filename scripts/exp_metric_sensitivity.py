"""T2.2 -- Metric-sensitivity sweep (answers reviewer W: "recover metric feels
hand-designed", "why 50 steps", and the request to pin down the audit).

Post-hoc on existing checkpoints (no retraining):
  1. Relearning recovery vs finetune budget: sweep steps in {10,25,50,100,200}
     (and optionally lr) and report person_IoU after few-shot person relearning.
     If ADVR stays near the oracle across the WHOLE curve, recover_pts is not an
     artifact of the 50-step choice.
  2. Threshold-free leakage metrics (residual_person, blend_fail, silhouette_IoU)
     recomputed so all numbers come from one script.

blend_fail is defined precisely in seg/audit.py: the % of person-region pixels whose
argmax prediction differs from the dominant-neighbour target (i.e. the region did
NOT blend in). This script emits it alongside the sweep so the paper can cite a
single reproducible source.

Usage: PYTHONPATH=. python scripts/exp_metric_sensitivity.py
Env: MODELS (default "oracle,ADVR,bl_SSD,bl_DELETE_s1"), STEPS, ARCH, OUT_JSON
"""
import json
import os

import torch
from torch.utils.data import DataLoader

from seg.audit import leakage_metrics, relearn_audit
from seg.data import SegDS, UnlearnDS
from seg.engine import seg_metrics
from seg.model import get_seg_model

STEPS = [int(s) for s in os.environ.get("STEPS", "10,25,50,100,200").split(",")]
MODELS = [m for m in os.environ.get("MODELS",
          "oracle,ADVR,bl_SSD_s1,bl_DELETE_s1").split(",") if m]


def _resolve(name, arch, cls):
    if name in ("oracle", "original"):
        return f"checkpoints/seg_{name}_{arch}_cls{cls}.pth"
    return f"checkpoints/unlearned_{name}.pth"


def main():
    arch = os.environ.get("ARCH", "deeplabv3r50")
    cls = os.environ.get("FORGET_CLS", "12")
    device = "cuda"
    vl = DataLoader(SegDS("val", "orig", size=512, crop=False), batch_size=4,
                    num_workers=8, pin_memory=True)
    uvl = DataLoader(UnlearnDS("val", size=512, crop=False), batch_size=4,
                     num_workers=8, pin_memory=True)
    ul = DataLoader(UnlearnDS("train", size=512, crop=True), batch_size=4,
                    shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    rows = {}
    for name in MODELS:
        path = _resolve(name, arch, cls)
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} missing"); continue
        ck = torch.load(path, map_location=device, weights_only=False)
        m = get_seg_model(device=device, arch=arch); m.load_state_dict(ck["state_dict"])
        base = seg_metrics(m, vl, device)
        leak = leakage_metrics(m, uvl, device)
        curve = {}
        for s in STEPS:
            au = relearn_audit(m, ul, device=device, steps=s, val_loader=vl)
            curve[s] = {"person_IoU_after": round(au["person_IoU_after"], 1),
                        "recover_pts": au.get("recover_pts")}
            print(f"  {name:14s} steps={s:3d}: person_IoU_after "
                  f"{curve[s]['person_IoU_after']:.1f} (recover {curve[s]['recover_pts']})",
                  flush=True)
        rows[name] = {
            "person_IoU": round(base["person_IoU"], 1),
            "retain_mIoU": round(base["retain_mIoU"], 1),
            "residual_person": round(leak["residual_person"], 1),
            "blend_fail": round(leak["blend_fail"], 1),
            "silhouette_IoU": round(leak["silhouette_IoU"], 1),
            "recover_curve": curve,
        }
        del m; torch.cuda.empty_cache()

    out = os.environ.get("OUT_JSON", "results/seg/exp_metric_sensitivity.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    print("recover_curve shows person_IoU vs relearning budget -- a method is "
          "robustly forgotten only if it stays low across ALL step counts.")


if __name__ == "__main__":
    main()
