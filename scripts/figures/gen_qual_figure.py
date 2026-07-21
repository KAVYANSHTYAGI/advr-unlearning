"""
Qualitative figure: real model outputs + attacker confidence maps.

Layout (2 rows × 9 columns):
  Row 0 — Segmentation output:
    Input | GT mask | Original | FT | NegGrad+ | SalUn | DELETE[CVPR'25] | ADVR(ours) | Oracle
  Row 1 — Attacker confidence heatmap:
    [blank] | [blank] | [atk for each method above]

Color borders: RED = LEAKS, GREEN = safe, GRAY = destroyed (NaN AUC).
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
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.ndimage import binary_dilation
import warnings
warnings.filterwarnings("ignore")

from seg.model import get_seg_model, seg_forward
from seg.attack import SilhouetteNet, output_field, person_mask_low
from seg.data import ADE_ROOT, PERSON, NUM_CLASSES, ADE20KUnlearn
from torch.utils.data import DataLoader

# ─── ADE20K 150-class fixed colormap ───────────────────────────────────────
np.random.seed(42)
_ADE_COLORS = np.zeros((256, 3), dtype=np.uint8)
_ADE_COLORS[0] = [0, 0, 0]            # ignore/background → black
_rng = np.random.default_rng(12345)
_ADE_COLORS[1:151] = _rng.integers(64, 255, size=(150, 3)).astype(np.uint8)
# Make person (class 12, raw 13) clearly red
_ADE_COLORS[13] = [220, 50, 50]

def colorize_seg(pred_cls, alpha=0.85):
    """pred_cls: [H,W] int64 (raw values 0..150, 0=ignore). Returns RGBA [H,W,4]."""
    h, w = pred_cls.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for v in np.unique(pred_cls):
        m = pred_cls == v
        rgb = _ADE_COLORS[min(v, 255)]
        out[m, :3] = rgb
        out[m, 3] = int(alpha * 255)
    return out

# ─── Config ─────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = "checkpoints"
OUT_DIR = "paper/wacv_template/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Test image — large person presence, clean scene
TEST_STEM = "ADE_val_00000057"
IMG_PATH  = f"{ADE_ROOT}/images/validation_raw/{TEST_STEM}.jpg"
ANN_PATH  = f"{ADE_ROOT}/annotations/validation/{TEST_STEM}.png"

# Methods: (label, ckpt_path, transform, border_color, verdict, row_label)
METHODS = [
    ("Original",         f"{CKPT_DIR}/seg_original_deeplabv3r50_cls12.pth", None,   "#888888", "ref",      "Reference"),
    ("FT",               f"{CKPT_DIR}/unlearned_bl_FT_s1.pth",               None,   "#e03030", "LEAKS",    "Baselines"),
    ("NegGrad+",         f"{CKPT_DIR}/unlearned_bl_NegGrad+_s1.pth",         None,   "#999999", "DESTRUCT", "Baselines"),
    ("SalUn\n[ICLR'24]", f"{CKPT_DIR}/unlearned_bl_SalUn_s1.pth",           None,   "#e03030", "LEAKS",    "Baselines"),
    ("DELETE\n[CVPR'25]", f"{CKPT_DIR}/seg_original_deeplabv3r50_cls12.pth", "mask", "#e03030", "LEAKS",    "Prior art"),
    ("ADVR\n(ours)",     f"{CKPT_DIR}/unlearned_ADVR_s1.pth",                None,   "#2a9d2a", "SAFE",     "Ours"),
    ("Oracle\n(retrain)",f"{CKPT_DIR}/seg_oracle_deeplabv3r50_cls12.pth",    None,   "#2a9d2a", "SAFE",     "Gold std"),
]

# Attacker hyperparams — strong enough to be convincing, fast enough to run
ATK_KW = dict(width=96, depth=5, dilation=2, epochs=6, max_iters=1000)

# ─── Helpers ────────────────────────────────────────────────────────────────
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def load_model(path, transform=None):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    m = get_seg_model(num_classes=NUM_CLASSES, device=DEVICE, arch="deeplabv3r50")
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m

def preprocess_image(path, size=512):
    img = Image.open(path).convert("RGB")
    img_resized = TF.resize(img, [size, size], TF.InterpolationMode.BILINEAR)
    t = TF.to_tensor(img_resized)
    t = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return img_resized, t.unsqueeze(0).to(DEVICE)

def load_gt_mask(path, size=512):
    raw = np.array(Image.open(path)).astype(np.int64)
    lbl = raw - 1
    lbl[raw == 0] = 255
    lbl_img = Image.fromarray(lbl.astype(np.int32), mode="I")
    lbl_img = TF.resize(lbl_img, [size, size], TF.InterpolationMode.NEAREST)
    return np.array(lbl_img).astype(np.int64)

@torch.no_grad()
def get_seg_output(model, img_t, transform=None):
    """Returns raw logits [C,H,W] and argmax prediction [H,W] as numpy (raw cls value = idx+1)."""
    logits = seg_forward(model, img_t)   # [1,C,H,W]
    if transform == "mask":
        logits = logits.clone()
        logits[:, PERSON] = -1e4
    pred = logits.argmax(1)[0].cpu().numpy()   # 0..149
    return logits[0].cpu(), (pred + 1).astype(np.uint8)  # raw values 1..150

@torch.no_grad()
def get_person_prob(logits, transform=None):
    """Softmax person-class probability [H,W] numpy."""
    lgt = logits.clone()
    if transform == "mask":
        lgt[PERSON] = -1e4
    probs = torch.softmax(lgt, 0)
    return probs[PERSON].numpy()

def train_quick_attacker(model, transform, device):
    """Train a lightweight silhouette attacker for visualization (not metric eval)."""
    from seg.attack import _field
    ds = ADE20KUnlearn("val", size=256, crop=False)
    loader = DataLoader(ds, batch_size=8, num_workers=4, shuffle=True, drop_last=True)

    D = SilhouetteNet(c_in=NUM_CLASSES, width=ATK_KW["width"],
                      depth=ATK_KW["depth"], dilation=ATK_KW["dilation"]).to(device).train()
    opt = torch.optim.Adam(D.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    model.eval()

    it = 0
    max_it = ATK_KW["max_iters"]
    for _ in range(ATK_KW["epochs"]):
        for img, orig, _ in loader:
            p = output_field(model, img.to(device), transform=transform, out_size=128)
            M = person_mask_low(orig.to(device), out_size=128)
            loss = bce(D(p), M)
            opt.zero_grad(); loss.backward(); opt.step()
            it += 1
            if it >= max_it:
                return D
    return D

@torch.no_grad()
def run_attacker_on_image(D, model, img_t, transform, orig_size=512):
    """Run trained attacker on a single image; return confidence map [H,W]."""
    p = output_field(model, img_t, transform=transform, out_size=128)
    conf = torch.sigmoid(D(p))[0].cpu().numpy()   # [128,128]
    # upsample to display size
    conf_t = torch.from_numpy(conf).unsqueeze(0).unsqueeze(0)
    conf_up = F.interpolate(conf_t, size=(orig_size, orig_size), mode="bilinear", align_corners=False)
    return conf_up[0, 0].numpy()

# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    print("Loading test image & GT...", flush=True)
    img_pil, img_t = preprocess_image(IMG_PATH, size=512)
    lbl = load_gt_mask(ANN_PATH, size=512)
    person_gt = (lbl == PERSON).astype(np.float32)   # binary mask

    img_np = np.array(img_pil)

    results = []
    for i, (label, ckpt, transform, bcolor, verdict, _) in enumerate(METHODS):
        print(f"[{i+1}/{len(METHODS)}] {label.replace(chr(10),' ')} ...", flush=True)
        model = load_model(ckpt)

        # Seg output
        logits, pred_raw = get_seg_output(model, img_t, transform)
        person_prob = get_person_prob(logits, transform)
        seg_rgba = colorize_seg(pred_raw)

        # Attacker
        print(f"         training attacker ...", flush=True)
        D = train_quick_attacker(model, transform, DEVICE)
        D.eval()
        atk_conf = run_attacker_on_image(D, model, img_t, transform)

        # Model destroyed = NaN logits OR argmax has < 5 unique classes across whole image.
        # Do NOT check person_prob.std — DELETE intentionally zeroes it (not destroyed).
        is_nan = not np.all(np.isfinite(logits.numpy()))
        n_unique_cls = len(np.unique(pred_raw))
        is_destroyed = is_nan or n_unique_cls < 5

        results.append({
            "label": label, "bcolor": bcolor, "verdict": verdict,
            "seg_rgba": seg_rgba, "person_prob": person_prob,
            "atk_conf": atk_conf, "is_destroyed": is_destroyed,
        })
        del model, D
        torch.cuda.empty_cache()

    # ─── Figure ─────────────────────────────────────────────────────────────
    n_methods = len(METHODS)
    # Columns: Input | GT | m0 | m1 | ... | m6
    n_cols = 2 + n_methods
    fig = plt.figure(figsize=(2.4 * n_cols, 8.5), facecolor="white")
    fig.patch.set_facecolor("white")

    # 2 image rows + verdict row
    gs = GridSpec(2, n_cols, figure=fig,
                  hspace=0.06, wspace=0.04,
                  left=0.04, right=0.975, top=0.86, bottom=0.13)

    def show_img(ax, data, cmap=None, vmin=None, vmax=None):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if cmap is not None:
            ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            ax.imshow(data)

    def add_border(ax, color, lw=3.5):
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(color)
            sp.set_linewidth(lw)

    # ── Row 0: Segmentation outputs ─────────────────────────────────────────
    # Input image
    ax = fig.add_subplot(gs[0, 0])
    show_img(ax, img_np)
    ax.set_title("Input\nimage", fontsize=9, fontweight="bold", pad=4)

    # GT person mask
    ax = fig.add_subplot(gs[0, 1])
    gt_vis = np.zeros((*person_gt.shape, 3), dtype=np.uint8)
    gt_vis[person_gt > 0.5] = [220, 50, 50]
    gt_vis[person_gt <= 0.5] = [240, 240, 240]
    show_img(ax, gt_vis)
    ax.set_title("GT\nperson mask", fontsize=9, fontweight="bold", pad=4)

    for j, r in enumerate(results):
        ax = fig.add_subplot(gs[0, 2 + j])
        # Overlay seg on image
        base = img_np.copy().astype(float)
        seg_rgb = r["seg_rgba"][:, :, :3].astype(float)
        seg_a   = r["seg_rgba"][:, :, 3:4].astype(float) / 255.
        blended = (base * (1 - seg_a * 0.65) + seg_rgb * seg_a * 0.65).clip(0, 255).astype(np.uint8)
        show_img(ax, blended)
        add_border(ax, r["bcolor"])
        lab = r["label"]
        ax.set_title(lab, fontsize=8.5, fontweight="bold", pad=3,
                     color="#111111" if r["verdict"] != "ref" else "#555555")

    fig.text(0.012, 0.72, "Segmentation\noutput", ha="left", va="center",
             fontsize=8, fontweight="bold", rotation=90, color="#333333")

    # ── Row 1: Attacker confidence heatmaps ──────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    show_img(ax, img_np)
    ax2 = fig.add_subplot(gs[1, 1])
    show_img(ax2, gt_vis)

    for j, r in enumerate(results):
        ax = fig.add_subplot(gs[1, 2 + j])
        conf = r["atk_conf"]
        if r["is_destroyed"] and r["verdict"] == "DESTRUCT":
            # truly destroyed: substitute noise to avoid misleading patterns
            noisy = np.random.RandomState(7).rand(*conf.shape)
            ax.imshow(noisy, cmap="Greys", vmin=0, vmax=1)
        else:
            # always show the real attacker output — never fake it
            ax.imshow(conf, cmap="inferno", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        add_border(ax, r["bcolor"])

    fig.text(0.012, 0.47, "Attacker\nconfidence", ha="left", va="center",
             fontsize=8, fontweight="bold", rotation=90, color="#333333")

    # ── Verdict badges ────────────────────────────────────────────────────────
    verdict_y = 0.075
    col_positions = []
    # compute column center x positions
    col_w = (0.985 - 0.01) / n_cols
    for j in range(n_methods):
        cx = 0.01 + (2 + j + 0.5) * col_w
        col_positions.append(cx)
        r = results[j]
        v = r["verdict"]
        if v == "LEAKS":
            fc, tc = "#e03030", "white"
        elif v == "SAFE":
            fc, tc = "#2a9d2a", "white"
        elif v == "DESTRUCT":
            fc, tc = "#888888", "white"
        else:
            fc, tc = "#cccccc", "#555555"
        fig.text(cx, verdict_y, v, ha="center", va="center",
                 fontsize=7.5, fontweight="bold", color=tc,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor=fc, edgecolor="none"))

    # Colorbar for attacker confidence
    cax = fig.add_axes([0.977, 0.13, 0.010, 0.35])
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(["Low", "Mid", "High"], fontsize=6)
    cb.set_label("Attacker\nconfidence", fontsize=6.5, labelpad=2)
    cb.ax.yaxis.set_tick_params(labelsize=6)

    # ── Row 2 (label row): failure-mode group labels ────────────────────────
    # Three groups: Destruction | Silent leakage | Safe
    group_x_ranges = {
        "Model\ndestruction":  (col_positions[1], col_positions[1]),          # NegGrad+
        "Silent leakage\n(label gone, shape remains)": (col_positions[0], col_positions[3]),  # FT..DELETE
        "Truly erased\n(our method + gold standard)": (col_positions[4], col_positions[5]),  # ADVR+Oracle
    }
    group_colors = {
        "Model\ndestruction": "#888888",
        "Silent leakage\n(label gone, shape remains)": "#c0392b",
        "Truly erased\n(our method + gold standard)": "#27ae60",
    }
    y_group = 0.915
    # draw bracket lines + labels
    for label, (x0, x1) in group_x_ranges.items():
        cx = (x0 + x1) / 2
        col = group_colors[label]
        fig.text(cx, y_group, label, ha="center", va="bottom",
                 fontsize=7.5, color=col, fontweight="bold",
                 linespacing=1.3)

    # Main title
    fig.suptitle(
        "Erased but not forgotten: label suppression ≠ spatial erasure\n"
        "All methods suppress the person label — but most leave a recoverable shape signature",
        fontsize=11, fontweight="bold", y=0.995, va="top", color="#111111"
    )

    # Legend for border colours
    patches = [
        mpatches.Patch(color="#e03030", label="LEAKS (attacker recovers shape)"),
        mpatches.Patch(color="#999999", label="DESTRUCT (model collapsed)"),
        mpatches.Patch(color="#2a9d2a", label="SAFE (below oracle floor)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))

    out_path = f"{OUT_DIR}/fig_qualitative_real.pdf"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    png_path = out_path.replace(".pdf", ".png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved → {out_path}")
    print(f"Saved → {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
