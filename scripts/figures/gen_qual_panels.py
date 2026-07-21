"""
Save every panel as its own individual PNG — no layout composition.
LaTeX handles the grid.

Output: paper/wacv_template/figures/qual/
  input.png
  gt_mask.png
  seg_<name>.png   — segmentation overlay on real image
  atk_<name>.png   — attacker confidence heatmap (inferno)

Names: original  ft  neggrad+  salun  delete  advr  oracle
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
from seg.attack import SilhouetteNet, output_field, person_mask_low
from seg.data import ADE_ROOT, PERSON, NUM_CLASSES, ADE20KUnlearn
from torch.utils.data import DataLoader

# ── ADE20K colormap (fixed seed) ────────────────────────────────────────────
np.random.seed(42)
_ADE_COLORS = np.zeros((256, 3), dtype=np.uint8)
_rng = np.random.default_rng(12345)
_ADE_COLORS[1:151] = _rng.integers(64, 230, size=(150, 3)).astype(np.uint8)
_ADE_COLORS[13] = [220, 50, 50]   # person → red

# ── Config ───────────────────────────────────────────────────────────────────
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
CKPT     = "checkpoints"
OUT_DIR  = "paper/wacv_template/figures/qual"
os.makedirs(OUT_DIR, exist_ok=True)

TEST_STEM = "ADE_val_00000057"
IMG_PATH  = f"{ADE_ROOT}/images/validation_raw/{TEST_STEM}.jpg"
ANN_PATH  = f"{ADE_ROOT}/annotations/validation/{TEST_STEM}.png"
SZ = 512

# (slug, label, ckpt, transform)
METHODS = [
    ("original",  "Original model",     f"{CKPT}/seg_original_deeplabv3r50_cls12.pth", None),
    ("ft",        "FT",                  f"{CKPT}/unlearned_bl_FT_s1.pth",               None),
    ("neggrad+",  "NegGrad+",            f"{CKPT}/unlearned_bl_NegGrad+_s1.pth",         None),
    ("salun",     "SalUn [ICLR'24]",    f"{CKPT}/unlearned_bl_SalUn_s1.pth",            None),
    ("delete",    "DELETE [CVPR'25]",   f"{CKPT}/seg_original_deeplabv3r50_cls12.pth",  "mask"),
    ("advr",      "ADVR (ours)",         f"{CKPT}/unlearned_ADVR_s1.pth",                None),
    ("oracle",    "Oracle (retrain)",    f"{CKPT}/seg_oracle_deeplabv3r50_cls12.pth",    None),
]

ATK_KW = dict(width=96, depth=5, dilation=2, epochs=6, max_iters=1000)

# ── Helpers ──────────────────────────────────────────────────────────────────
def preprocess(path):
    img = Image.open(path).convert("RGB")
    img_r = TF.resize(img, [SZ, SZ], TF.InterpolationMode.BILINEAR)
    t = TF.normalize(TF.to_tensor(img_r), [0.485,0.456,0.406], [0.229,0.224,0.225])
    return np.array(img_r), t.unsqueeze(0).to(DEVICE)

def load_gt(path):
    raw = np.array(Image.open(path)).astype(np.int64)
    lbl = raw - 1; lbl[raw == 0] = 255
    lbl_i = Image.fromarray(lbl.astype(np.int32), mode="I")
    lbl_i = TF.resize(lbl_i, [SZ, SZ], TF.InterpolationMode.NEAREST)
    return np.array(lbl_i).astype(np.int64)

def load_model(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    m = get_seg_model(num_classes=NUM_CLASSES, device=DEVICE, arch="deeplabv3r50")
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m

@torch.no_grad()
def forward(model, img_t, transform=None):
    logits = seg_forward(model, img_t)
    if transform == "mask":
        logits = logits.clone(); logits[:, PERSON] = -1e4
    pred = logits.argmax(1)[0].cpu().numpy()
    return logits[0].cpu(), (pred + 1).astype(np.uint8)

def seg_overlay(img_np, pred_raw, alpha=0.62):
    h, w = pred_raw.shape
    seg_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for v in np.unique(pred_raw):
        seg_rgb[pred_raw == v] = _ADE_COLORS[min(int(v), 255)]
    base  = img_np.astype(float)
    blend = (base * (1 - alpha) + seg_rgb.astype(float) * alpha).clip(0, 255).astype(np.uint8)
    return blend

def train_attacker(model, transform):
    ds     = ADE20KUnlearn("val", size=256, crop=False)
    loader = DataLoader(ds, batch_size=8, num_workers=4, shuffle=True, drop_last=True)
    D = SilhouetteNet(c_in=NUM_CLASSES, width=ATK_KW["width"],
                      depth=ATK_KW["depth"], dilation=ATK_KW["dilation"]).to(DEVICE).train()
    opt = torch.optim.Adam(D.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    it = 0
    for _ in range(ATK_KW["epochs"]):
        for img, orig, _ in loader:
            p = output_field(model, img.to(DEVICE), transform=transform, out_size=128)
            M = person_mask_low(orig.to(DEVICE), out_size=128)
            opt.zero_grad(); bce(D(p), M).backward(); opt.step()
            it += 1
            if it >= ATK_KW["max_iters"]: return D
    return D

@torch.no_grad()
def atk_heatmap(D, model, img_t, transform):
    p    = output_field(model, img_t, transform=transform, out_size=128)
    conf = torch.sigmoid(D(p))[0].cpu().numpy()
    conf_t = torch.from_numpy(conf).unsqueeze(0).unsqueeze(0)
    return F.interpolate(conf_t, size=(SZ, SZ), mode="bilinear",
                         align_corners=False)[0, 0].numpy()

def save_img(arr_hwc, path, dpi=200):
    """Save an RGB numpy array as a clean borderless PNG."""
    h, w = arr_hwc.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(w/dpi, h/dpi), dpi=dpi)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.imshow(arr_hwc); ax.axis("off")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  saved {path}")

def save_heatmap(conf_hw, path, dpi=200, cmap="inferno"):
    h, w = conf_hw.shape
    fig, ax = plt.subplots(1, 1, figsize=(w/dpi, h/dpi), dpi=dpi)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.imshow(conf_hw, cmap=cmap, vmin=0, vmax=1); ax.axis("off")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  saved {path}")

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading image & GT...")
    img_np, img_t = preprocess(IMG_PATH)
    lbl = load_gt(ANN_PATH)
    person_gt = (lbl == PERSON).astype(np.uint8)

    # ── Input image ──
    save_img(img_np, f"{OUT_DIR}/input.png")

    # ── GT mask ──
    gt_rgb = np.zeros((SZ, SZ, 3), dtype=np.uint8)
    gt_rgb[person_gt > 0] = [220, 50, 50]
    gt_rgb[person_gt == 0] = [245, 245, 245]
    save_img(gt_rgb, f"{OUT_DIR}/gt_mask.png")

    for slug, label, ckpt_path, transform in METHODS:
        print(f"\n[{slug}] {label}")
        model = load_model(ckpt_path)

        # seg overlay
        _, pred_raw = forward(model, img_t, transform)
        overlay = seg_overlay(img_np, pred_raw)
        save_img(overlay, f"{OUT_DIR}/seg_{slug}.png")

        # attacker heatmap
        print(f"  training attacker...")
        D = train_attacker(model, transform); D.eval()
        conf = atk_heatmap(D, model, img_t, transform)
        save_heatmap(conf, f"{OUT_DIR}/atk_{slug}.png")

        del model, D; torch.cuda.empty_cache()

    print("\nAll panels saved to", OUT_DIR)

if __name__ == "__main__":
    main()
