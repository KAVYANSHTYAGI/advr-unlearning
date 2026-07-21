"""Clean eval of a saved unlearned checkpoint (fresh process, no training memory)."""
import argparse
import torch
from torch.utils.data import DataLoader

from seg.audit import leakage_metrics, relearn_audit
from seg.data import SegDS, UnlearnDS
from seg.engine import seg_metrics
from seg.model import get_seg_model

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--arch", default="deeplabv3r50")
args = ap.parse_args(); device = "cuda"

vl = DataLoader(SegDS("val", "orig", size=512, crop=False), batch_size=4,
                num_workers=8, pin_memory=True)
uvl = DataLoader(UnlearnDS("val", size=512, crop=False), batch_size=4,
                 num_workers=8, pin_memory=True)
ul = DataLoader(UnlearnDS("train", size=512, crop=True), batch_size=4, shuffle=True,
                num_workers=8, pin_memory=True, drop_last=True)

ck = torch.load(f"checkpoints/unlearned_{args.name}.pth", map_location=device, weights_only=False)
m = get_seg_model(device=device, arch=args.arch); m.load_state_dict(ck["state_dict"])
r = seg_metrics(m, vl, device); r.update(leakage_metrics(m, uvl, device))
au = relearn_audit(m, ul, device=device, steps=50, val_loader=vl)
print(f"{args.name}: person {r['person_IoU']:.1f} | retain {r['retain_mIoU']:.1f} "
      f"| blend_fail {r['blend_fail']:.1f} | silhouette {r['silhouette_IoU']:.1f} "
      f"| recover {au.get('recover_pts')}")
