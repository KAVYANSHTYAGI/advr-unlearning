"""
Individual panel PNGs for Cityscapes qualitative figure.
Scene: Frankfurt pedestrian crossing (frankfurt_000001_013016) — clear individual
       person silhouettes against a clean urban street backdrop.

Output: paper/wacv_template/figures/qual_city/
  input.png  gt_mask.png
  seg_<name>.png   atk_<name>.png
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["DATASET"]    = "cityscapes"
os.environ["FORGET_CLS"] = "11"
os.environ["NUM_CLASSES"] = "19"

import importlib
import seg.data as _sd
importlib.reload(_sd)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from seg.model import get_seg_model, seg_forward
from seg.attack import output_field, person_mask_low
from seg.data import (CITY_ROOT, _CITY_LABELID_TO_TRAINID,
                      CityscapesUnlearn, merge_person_label)
import torch.nn as nn

# ── Cityscapes 19-class standard palette (trainId → RGB) ────────────────────
_CITY_PALETTE = np.array([
    [128,  64, 128],   #  0 road
    [244,  35, 232],   #  1 sidewalk
    [ 70,  70,  70],   #  2 building
    [102, 102, 156],   #  3 wall
    [190, 153, 153],   #  4 fence
    [153, 153, 153],   #  5 pole
    [250, 170,  30],   #  6 traffic light
    [220, 220,   0],   #  7 traffic sign
    [107, 142,  35],   #  8 vegetation
    [152, 251, 152],   #  9 terrain
    [ 70, 130, 180],   # 10 sky
    [220,  20,  60],   # 11 person  ← forget class (vivid red)
    [255,   0,   0],   # 12 rider
    [  0,   0, 142],   # 13 car
    [  0,   0,  70],   # 14 truck
    [  0,  60, 100],   # 15 bus
    [  0,  80, 100],   # 16 train
    [  0,   0, 230],   # 17 motorcycle
    [119,  11,  32],   # 18 bicycle
], dtype=np.uint8)

# ── Config ───────────────────────────────────────────────────────────────────
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
CKPT     = "checkpoints"
OUT_DIR  = "paper/wacv_template/figures/qual_city"
os.makedirs(OUT_DIR, exist_ok=True)

PERSON_TRAINID = 11   # Cityscapes person trainId
NUM_CLS        = 19
SZ             = 512

CITY   = "frankfurt"
STEM   = "frankfurt_000001_013016"
IMG_PATH = f"{CITY_ROOT}/leftImg8bit/val/{CITY}/{STEM}_leftImg8bit.png"
ANN_PATH = f"{CITY_ROOT}/gtFine/val/{CITY}/{STEM}_gtFine_labelIds.png"

# (slug, label, ckpt, transform)
METHODS = [
    ("original",  "Original model",   f"{CKPT}/seg_original_deeplabv3r50_cls11.pth",  None),
    ("ft",        "FT",               f"{CKPT}/unlearned_city_bl_FT_s1.pth",           None),
    ("neggrad+",  "NegGrad+",         f"{CKPT}/unlearned_city_bl_NegGrad+_s1.pth",     None),
    ("salun",     "SalUn [ICLR'24]", f"{CKPT}/unlearned_city_bl_SalUn_s1.pth",        None),
    ("delete",    "DELETE [CVPR'25]", f"{CKPT}/seg_original_deeplabv3r50_cls11.pth",  "mask"),
    ("advr",      "ADVR (ours)",      f"{CKPT}/unlearned_city_ADVR_s1.pth",            None),
    ("oracle",    "Oracle (retrain)", f"{CKPT}/seg_oracle_deeplabv3r50_cls11.pth",     None),
]

ATK_KW = dict(width=96, depth=5, dilation=2, epochs=6, max_iters=1000)

# ── Helpers ──────────────────────────────────────────────────────────────────
def preprocess(path):
    img = Image.open(path).convert("RGB")
    # Crop to centre-square to get a clean framing of the crossing
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    img_c = img.crop((left, 0, left + side, side))
    img_r = TF.resize(img_c, [SZ, SZ], TF.InterpolationMode.BILINEAR)
    t = TF.normalize(TF.to_tensor(img_r), [0.485,0.456,0.406], [0.229,0.224,0.225])
    return np.array(img_r), t.unsqueeze(0).to(DEVICE), (left, 0, left+side, side)

def load_gt(path, crop_box):
    raw = np.array(Image.open(path))
    train = _CITY_LABELID_TO_TRAINID[raw].astype(np.int64)
    # apply same crop
    l, t, r, b = crop_box
    train = train[t:b, l:r]
    lbl_i = Image.fromarray(train.astype(np.int32), mode="I")
    lbl_i = TF.resize(lbl_i, [SZ, SZ], TF.InterpolationMode.NEAREST)
    return np.array(lbl_i).astype(np.int64)

def load_model(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    m = get_seg_model(num_classes=NUM_CLS, device=DEVICE, arch="deeplabv3r50")
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m

@torch.no_grad()
def forward(model, img_t, transform=None):
    logits = seg_forward(model, img_t)             # [1,19,H,W]
    if transform == "mask":
        logits = logits.clone()
        logits[:, PERSON_TRAINID] = -1e4
    pred = logits.argmax(1)[0].cpu().numpy()       # 0..18 trainId
    return logits[0].cpu(), pred.astype(np.uint8)

def seg_overlay(img_np, pred, alpha=0.60):
    h, w = pred.shape
    seg_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for v in np.unique(pred):
        seg_rgb[pred == v] = _CITY_PALETTE[int(v) % 19]
    base  = img_np.astype(float)
    blend = (base*(1-alpha) + seg_rgb.astype(float)*alpha).clip(0,255).astype(np.uint8)
    return blend

# Lightweight SilhouetteNet (same as seg/attack.py but standalone to avoid env issues)
class _SNet(nn.Module):
    def __init__(self, c_in, width, depth, dilation):
        super().__init__()
        layers, cin = [], c_in
        for _ in range(depth):
            layers += [nn.Conv2d(cin, width, 3, padding=dilation, dilation=dilation),
                       nn.BatchNorm2d(width), nn.ReLU(True)]
            cin = width
        layers += [nn.Conv2d(width, 1, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, p):
        return self.net(p)[:, 0]

def _out_field(model, img, transform, out_size=128):
    with torch.no_grad():
        z = seg_forward(model, img)
        if transform == "mask":
            z = z.clone(); z[:, PERSON_TRAINID] = -1e4
        p = F.softmax(z, 1)
        return F.interpolate(p, (out_size, out_size), mode="bilinear", align_corners=False)

def _person_mask(lbl, out_size=128):
    M = (lbl == PERSON_TRAINID).float().unsqueeze(1)
    return F.interpolate(M, (out_size, out_size), mode="nearest")[:, 0]

def train_attacker(model, transform):
    ds = CityscapesUnlearn("val", size=256, crop=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=4,
                                          shuffle=True, drop_last=True)
    D = _SNet(c_in=NUM_CLS, width=ATK_KW["width"],
              depth=ATK_KW["depth"], dilation=ATK_KW["dilation"]).to(DEVICE).train()
    opt = torch.optim.Adam(D.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    it = 0
    for _ in range(ATK_KW["epochs"]):
        for img, orig, _ in loader:
            p = _out_field(model, img.to(DEVICE), transform)
            M = _person_mask(orig.to(DEVICE))
            opt.zero_grad(); bce(D(p), M).backward(); opt.step()
            it += 1
            if it >= ATK_KW["max_iters"]: return D
    return D

@torch.no_grad()
def atk_heatmap(D, model, img_t, transform):
    p    = _out_field(model, img_t, transform)
    conf = torch.sigmoid(D(p))[0].cpu().numpy()
    ct   = torch.from_numpy(conf).unsqueeze(0).unsqueeze(0)
    return F.interpolate(ct, (SZ, SZ), mode="bilinear", align_corners=False)[0,0].numpy()

def save_img(arr, path, dpi=200):
    h, w = arr.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(w/dpi, h/dpi), dpi=dpi)
    fig.subplots_adjust(0,0,1,1)
    ax.imshow(arr); ax.axis("off")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  saved {path}")

def save_heatmap(conf, path, dpi=200):
    h, w = conf.shape
    fig, ax = plt.subplots(1, 1, figsize=(w/dpi, h/dpi), dpi=dpi)
    fig.subplots_adjust(0,0,1,1)
    ax.imshow(conf, cmap="inferno", vmin=0, vmax=1); ax.axis("off")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  saved {path}")

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading Cityscapes image & GT...", flush=True)
    img_np, img_t, crop_box = preprocess(IMG_PATH)
    lbl = load_gt(ANN_PATH, crop_box)

    save_img(img_np, f"{OUT_DIR}/input.png")

    gt_rgb = np.zeros((SZ, SZ, 3), dtype=np.uint8)
    gt_rgb[lbl == PERSON_TRAINID] = [220, 20, 60]
    gt_rgb[lbl != PERSON_TRAINID] = [245, 245, 245]
    save_img(gt_rgb, f"{OUT_DIR}/gt_mask.png")

    for slug, label, ckpt_path, transform in METHODS:
        print(f"\n[{slug}] {label}", flush=True)
        model = load_model(ckpt_path)

        _, pred = forward(model, img_t, transform)
        save_img(seg_overlay(img_np, pred), f"{OUT_DIR}/seg_{slug}.png")

        print("  training attacker...", flush=True)
        D = train_attacker(model, transform); D.eval()
        save_heatmap(atk_heatmap(D, model, img_t, transform), f"{OUT_DIR}/atk_{slug}.png")

        del model, D; torch.cuda.empty_cache()

    print(f"\nAll Cityscapes panels saved → {OUT_DIR}")

if __name__ == "__main__":
    main()
