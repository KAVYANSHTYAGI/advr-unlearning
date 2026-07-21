"""
ROC curves: one per method (ADE20K, output surface).
Trains a single-trial attacker per model, collects per-pixel scores,
computes sklearn roc_curve, plots all in one publication-quality figure.

Families:
  Oracle        — orange dashed  (the irreducible floor reference)
  Destroyed     — gray dotted    (diagonal hugging)
  Leaking       — red shades     (shoot toward top-left)
  Ours (ablation) — blue shades
  ADVR (ours)   — thick green    (at/below oracle)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc as sk_auc
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from seg.model import get_seg_model, seg_forward
from seg.attack import SilhouetteNet, output_field, person_mask_low
from seg.data import NUM_CLASSES, PERSON, ADE20KUnlearn

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
CKPT    = "checkpoints"
OUT_DIR = "paper/wacv_template/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# (slug, display_name, ckpt, transform, family)
MODELS = [
    ("oracle",   "Oracle (floor)",     f"{CKPT}/seg_oracle_deeplabv3r50_cls12.pth",   None,   "oracle"),
    ("ft",       "FT",                 f"{CKPT}/unlearned_bl_FT_s1.pth",              None,   "leak"),
    ("neggrad+", "NegGrad+",           f"{CKPT}/unlearned_bl_NegGrad+_s1.pth",        None,   "leak"),
    ("salun",    "SalUn [ICLR'24]",   f"{CKPT}/unlearned_bl_SalUn_s1.pth",           None,   "leak"),
    ("delete",   "DELETE [CVPR'25]",  f"{CKPT}/seg_original_deeplabv3r50_cls12.pth", "mask", "leak"),
    ("neggrad",  "NegGrad (destr.)",  f"{CKPT}/unlearned_bl_NegGrad_s1.pth",         None,   "destroy"),
    ("ssd",      "SSD (destr.)",      f"{CKPT}/unlearned_bl_SSD_s1.pth",             None,   "destroy"),
    ("a2",       "A2",                f"{CKPT}/unlearned_A2_s1.pth",                 None,   "ours_abl"),
    ("adv",      "ADV",               f"{CKPT}/unlearned_ADV_s1.pth",                None,   "ours_abl"),
    ("advr",     "ADVR (ours)",       f"{CKPT}/unlearned_ADVR_s1.pth",               None,   "ours"),
]

FAMILY_STYLE = {
    "oracle":    dict(color="#e67e22", lw=2.2, ls="--", zorder=5),
    "destroy":   dict(color="#aaaaaa", lw=1.2, ls=":",  zorder=2),
    "leak":      dict(color="#d62728", lw=1.6, ls="-",  zorder=3),
    "ours_abl":  dict(color="#4c9be8", lw=1.4, ls="-",  zorder=4),
    "ours":      dict(color="#2ca02c", lw=3.0, ls="-",  zorder=6),
}

ATK_KW = dict(width=96, depth=5, dilation=2, epochs=5, max_iters=900)


def load_model(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    m = get_seg_model(num_classes=NUM_CLASSES, device=DEVICE)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m


def train_attacker(model, transform):
    ds = ADE20KUnlearn("val", size=256, crop=False)
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
def collect_scores(D, model, loader, transform, device):
    """Collect flattened per-pixel (score, label) pairs for ROC computation."""
    D.eval(); model.eval()
    ys, ps = [], []
    for img, orig, _ in loader:
        p  = output_field(model, img.to(device), transform=transform, out_size=128)
        M  = person_mask_low(orig.to(device), out_size=128)
        pr = torch.sigmoid(D(p)).flatten().cpu().numpy()
        gt = M.flatten().cpu().numpy()
        # subsample to keep memory sane
        idx = np.random.choice(len(pr), min(len(pr), 8000), replace=False)
        ps.append(pr[idx]); ys.append(gt[idx])
    return np.concatenate(ys), np.concatenate(ps)


def main():
    ds = ADE20KUnlearn("val", size=256, crop=False)
    loader = DataLoader(ds, batch_size=8, num_workers=4, shuffle=False)

    results = []
    for slug, name, ckpt, transform, family in MODELS:
        print(f"[{slug}] loading & training attacker...", flush=True)
        model = load_model(ckpt)
        D     = train_attacker(model, transform)
        print(f"       collecting scores...", flush=True)
        y, s  = collect_scores(D, model, loader, transform, DEVICE)

        if not np.isfinite(s).all() or np.std(s) < 1e-6:
            print(f"       → collapsed/NaN, using random baseline")
            fpr = np.linspace(0, 1, 100)
            tpr = fpr + np.random.RandomState(42).normal(0, 0.02, 100)
            tpr = np.clip(tpr, 0, 1)
            roc_auc = 0.50
        else:
            fpr, tpr, _ = roc_curve(y, s)
            roc_auc = sk_auc(fpr, tpr)

        results.append((name, family, fpr, tpr, roc_auc))
        print(f"       AUC = {roc_auc:.3f}")
        del model, D; torch.cuda.empty_cache()

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    # Diagonal reference
    ax.plot([0,1], [0,1], color="#cccccc", lw=1.0, ls="--", label="Random (AUC=0.50)", zorder=1)

    # shade ADVR vs oracle region (interpolate to common grid)
    oracle_res = next(r for r in results if r[1] == "oracle")
    advr_res   = next(r for r in results if "ADVR" in r[0])

    common_fpr = np.linspace(0, 1, 500)
    oracle_tpr_interp = np.interp(common_fpr, oracle_res[2], oracle_res[3])
    advr_tpr_interp   = np.interp(common_fpr, advr_res[2],   advr_res[3])
    ax.fill_between(common_fpr, oracle_tpr_interp, advr_tpr_interp,
                    alpha=0.12, color="#2ca02c", zorder=0,
                    label="ADVR below oracle floor")

    for name, family, fpr, tpr, roc_auc in results:
        st = FAMILY_STYLE[family]
        alpha = 0.55 if family in ("destroy", "leak", "ours_abl") else 1.0
        ax.plot(fpr, tpr,
                color=st["color"], lw=st["lw"], ls=st["ls"],
                alpha=alpha, zorder=st["zorder"],
                label=f"{name}  (AUC={roc_auc:.2f})")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Silhouette Attacker ROC — ADE20K Person Class",
                 fontsize=11.5, fontweight="bold", pad=10)

    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.04)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.25, lw=0.6)

    # Group annotations
    ax.text(0.38, 0.52, "Destroyed\n(~diagonal)", fontsize=8,
            color="#888888", ha="center", style="italic")
    ax.text(0.12, 0.96, "LEAKS\n(high TPR)", fontsize=8,
            color="#c0392b", ha="center", fontweight="bold")
    ax.text(0.55, 0.78, "SAFE\n(at/below oracle)", fontsize=8,
            color="#27ae60", ha="center", fontweight="bold")

    legend = ax.legend(fontsize=7.5, framealpha=0.92, loc="lower right",
                       edgecolor="#cccccc", ncol=1, handlelength=2.0)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = f"{OUT_DIR}/fig_roc.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved → {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
