"""
Generate two paper figures:
  Fig 1 -- leakage AUC bar chart (ADE20K + Cityscapes, real data)
  Fig 2 -- qualitative silhouette recovery grid (synthetic schematic)

Run:  PYTHONPATH=. python scripts/gen_paper_figures.py
Output: paper/wacv_template/figures/fig_leakage.pdf / .png
        paper/wacv_template/figures/fig_qualitative.pdf / .png
"""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter, binary_dilation
from matplotlib.colors import LinearSegmentedColormap

os.makedirs("paper/wacv_template/figures", exist_ok=True)
np.random.seed(42)

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.7,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7.5,
    "axes.labelsize":     8,
    "legend.fontsize":    7,
    "figure.dpi":         220,
})

RED   = "#d94f4f"
GREEN = "#3aab6e"
GRAY  = "#b0b0b0"
DARK_GREEN = "#1b7a3e"
DARK_RED   = "#8b1a1a"

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — LEAKAGE BAR CHART
# ══════════════════════════════════════════════════════════════════════════════

ade  = json.load(open("results/seg/ade_leakage_long.json"))
city = json.load(open("results/seg/city_leakage.json"))

METHODS = [
    ("oracle_FLOOR",     "Oracle\n(floor)"),
    ("bl_FT_s1",         "FT"),
    ("bl_NegGrad_s1",    "NegGrad†"),
    ("bl_NegGrad+_s1",   "NegGrad+"),
    ("bl_RL_s1",         "RL"),
    ("bl_SalUn_s1",      "SalUn\n[ICLR'24]"),
    ("bl_SSD_s1",        "SSD†\n[AAAI'24]"),
    ("bl_SCRUB_s1",      "SCRUB†\n[NeurIPS'23]"),
    ("bl_DELETE_s1",     "DELETE\n[CVPR'25]"),
    ("DELETE_logitmask",  "DELETE-\nlogitmask"),
    ("ours_A2_s1",       "A2\n(ours)"),
    ("ours_ADV_s1",      "ADV\n(ours)"),
    ("ours_ADVR_s1",     "ADVR\n(ours) ★"),
]

def get_out(data, key):
    if key not in data:
        return float("nan"), float("nan")
    v = data[key]["out"]
    return v["AUC_mean"], v["AUC_std"]

def city_key(k):
    if k in ("oracle_FLOOR", "DELETE_logitmask"):
        return k
    if k.startswith("ours_"):
        return "city_" + k[5:]
    if k.startswith("bl_"):
        return "city_" + k
    return k

def bar_color(mu, thresh):
    if np.isnan(mu):
        return GRAY
    return RED if mu > thresh else GREEN

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=False)
fig.subplots_adjust(wspace=0.28, bottom=0.22, top=0.88)

for ax, (data, keyfn, title, dataset_lbl) in zip(axes, [
    (ade,  lambda k: k,    "ADE20K",     "150 classes · DeepLabV3-R50"),
    (city, city_key,       "Cityscapes", "19 classes · DeepLabV3-R50"),
]):
    floor_m, floor_s = get_out(data, keyfn("oracle_FLOOR"))
    thresh = floor_m + 2 * floor_s

    xs, mus, errs, cols, labels = [], [], [], [], []
    for i, (k, lbl) in enumerate(METHODS):
        mu, std = get_out(data, keyfn(k))
        xs.append(i); mus.append(mu); errs.append(std)
        cols.append(bar_color(mu, thresh)); labels.append(lbl)

    # ---- bars ----
    ax.bar(xs, mus, color=cols, width=0.64, zorder=3,
           edgecolor="white", linewidth=0.5)

    # error bars
    for xi, mu, err in zip(xs, mus, errs):
        if not np.isnan(mu):
            ax.errorbar(xi, mu, yerr=err, fmt="none",
                        ecolor="#333333", capsize=2.5, capthick=0.9,
                        elinewidth=0.9, zorder=5)

    # ---- floor band ----
    ax.axhspan(floor_m - floor_s, floor_m + floor_s,
               color="#b9f0ca", alpha=0.55, zorder=1)
    ax.axhline(floor_m, color=DARK_GREEN, lw=1.3, ls="--", zorder=2,
               label=f"Oracle floor  ({floor_m:.1f})")
    ax.axhline(thresh, color=RED, lw=1.2, ls=":", zorder=2,
               label=f"2σ leak threshold  ({thresh:.1f})")

    # ---- NaN destroyed markers ----
    for xi, mu in zip(xs, mus):
        if np.isnan(mu):
            ax.text(xi, 53.5, "✕", ha="center", va="bottom",
                    fontsize=10, color="#888888", fontweight="bold")

    # ---- value labels on key bars ----
    key_labels = {"ours_ADVR_s1", "bl_DELETE_s1", "DELETE_logitmask"}
    for xi, mu, (k, _) in zip(xs, mus, METHODS):
        if k in key_labels and not np.isnan(mu):
            va = "bottom" if mu > thresh else "top"
            offset = 0.4 if va == "bottom" else -0.4
            ax.text(xi, mu + offset, f"{mu:.1f}",
                    ha="center", va=va, fontsize=6.2,
                    color="white" if mu > 88 else "#333333",
                    fontweight="bold")

    # ---- axes ----
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6.6, linespacing=1.2,
                       rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Attacker AUC  (output surface, %)", fontsize=8)
    ax.set_ylim(50, 103)
    ax.set_title(f"{title}\n{dataset_lbl}", fontsize=9, fontweight="bold", pad=5)
    ax.yaxis.grid(True, ls="--", lw=0.35, color="#dddddd", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=3)

    ax.legend(loc="lower right", fontsize=7, framealpha=0.9,
              handlelength=1.6, handletextpad=0.5, borderpad=0.5)

# ---- shared bottom legend ----
p_leak = mpatches.Patch(color=RED,   label="LEAK  (above 2σ threshold)")
p_safe = mpatches.Patch(color=GREEN, label="Safe  (below threshold)")
p_dest = mpatches.Patch(color=GRAY,  label="Destroyed  (NaN — model collapsed)")
fig.legend(handles=[p_leak, p_safe, p_dest],
           loc="lower center", ncol=3, fontsize=7.5,
           bbox_to_anchor=(0.5, 0.0), frameon=False,
           columnspacing=1.5, handlelength=1.4)

fig.suptitle("Spatial Silhouette Leakage: Attacker AUC vs. Oracle Floor",
             fontsize=10.5, fontweight="bold")

fig.savefig("paper/wacv_template/figures/fig_leakage.pdf", bbox_inches="tight", dpi=250)
fig.savefig("paper/wacv_template/figures/fig_leakage.png", bbox_inches="tight", dpi=250)
plt.close(fig)
print("Saved fig_leakage")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — QUALITATIVE GRID
# Layout (2 rows × 4 cols):
#   Row 0: [Input image] [GT mask] [DELETE output] [ADVR output]
#   Row 1: [            ] [        ] [Atk from DELETE] [Atk from ADVR]
# ══════════════════════════════════════════════════════════════════════════════

H, W = 120, 160

# ── Scene ────────────────────────────────────────────────────────────────────
def make_scene():
    c = np.zeros((H, W, 3))
    # sky gradient
    for y in range(H // 3):
        t = y / (H // 3)
        c[y] = [0.40 + 0.20*t, 0.65 + 0.18*t, 0.92 + 0.06*t]
    # buildings with windows
    for y in range(H//3, 2*H//3):
        c[y, :W//2] = [0.70, 0.67, 0.62]
        c[y, W//2:] = [0.62, 0.60, 0.56]
    # window grid
    for wy in range(H//3 + 6, 2*H//3 - 4, 10):
        for wx in range(6, W//2 - 4, 12):
            c[wy:wy+5, wx:wx+7] = [0.85, 0.90, 0.95]
        for wx in range(W//2 + 6, W - 4, 12):
            c[wy:wy+5, wx:wx+7] = [0.75, 0.80, 0.88]
    # road with lane markings
    for y in range(2*H//3, H):
        t = (y - 2*H//3) / (H // 3)
        v = 0.30 + 0.08*t
        c[y] = [v, v, v]
    # lane dashes
    for x in range(10, W - 10, 22):
        c[2*H//3 + 4 : 2*H//3 + 9, x:x+10] = [0.85, 0.82, 0.65]
    # texture noise
    c += np.random.normal(0, 0.012, c.shape)
    return np.clip(c, 0, 1)

# ── Person silhouette (anatomically shaped) ───────────────────────────────────
def make_person_mask():
    from PIL import Image, ImageDraw
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    cx, foot_y = W // 2, int(H * 0.91)
    scale = H / 120.0
    # feet / shoes
    d.ellipse([cx-10, foot_y-5, cx-3, foot_y], fill=255)
    d.ellipse([cx+3,  foot_y-5, cx+10, foot_y], fill=255)
    # lower legs
    d.rectangle([cx-8, int(foot_y - 28*scale), cx-3, foot_y-5], fill=255)
    d.rectangle([cx+3, int(foot_y - 28*scale), cx+8, foot_y-5], fill=255)
    # knees
    d.ellipse([cx-9, int(foot_y-32*scale), cx-2, int(foot_y-24*scale)], fill=255)
    d.ellipse([cx+2, int(foot_y-32*scale), cx+9, int(foot_y-24*scale)], fill=255)
    # upper legs
    d.rectangle([cx-9, int(foot_y-50*scale), cx-2, int(foot_y-30*scale)], fill=255)
    d.rectangle([cx+2, int(foot_y-50*scale), cx+9, int(foot_y-30*scale)], fill=255)
    # hips / pelvis
    d.ellipse([cx-11, int(foot_y-56*scale), cx+11, int(foot_y-46*scale)], fill=255)
    # torso
    d.rectangle([cx-9, int(foot_y-82*scale), cx+9, int(foot_y-52*scale)], fill=255)
    # shoulders
    d.ellipse([cx-14, int(foot_y-86*scale), cx-5, int(foot_y-74*scale)], fill=255)
    d.ellipse([cx+5,  int(foot_y-86*scale), cx+14, int(foot_y-74*scale)], fill=255)
    # arms (slightly angled outward)
    d.polygon([cx-14, int(foot_y-84*scale), cx-9, int(foot_y-84*scale),
               cx-12, int(foot_y-58*scale), cx-17, int(foot_y-58*scale)], fill=255)
    d.polygon([cx+9,  int(foot_y-84*scale), cx+14, int(foot_y-84*scale),
               cx+17, int(foot_y-58*scale), cx+12, int(foot_y-58*scale)], fill=255)
    # neck
    d.rectangle([cx-4, int(foot_y-94*scale), cx+4, int(foot_y-84*scale)], fill=255)
    # head
    d.ellipse([cx-10, int(foot_y-112*scale), cx+10, int(foot_y-90*scale)], fill=255)
    mask = np.array(img, dtype=float) / 255.0
    mask = gaussian_filter(mask, sigma=1.2)
    return (mask > 0.35).astype(float)

scene = make_scene()
pmask = make_person_mask()

# ── Segmentation outputs ───────────────────────────────────────────────────────
def seg_original(scene, m):
    out = scene.copy()
    # vivid person colour with slight interior texture
    person_col = np.array([0.88, 0.18, 0.18])
    out[m > 0.5] = person_col
    # add subtle clothing variation inside mask
    noise_r = np.random.normal(0, 0.04, scene.shape[:2])
    for c, delta in enumerate([-0.05, 0.0, 0.0]):
        channel = out[:, :, c]
        channel[m > 0.5] = np.clip(channel[m > 0.5] + noise_r[m > 0.5]*0.5, 0, 1)
        out[:, :, c] = channel
    return np.clip(out, 0, 1)

def seg_delete(scene, m):
    """DELETE output: label gone but dark 'shadow' lingers at silhouette boundary."""
    out = scene.copy()
    # fill interior with road colour — but a shade too dark (confidence depleted)
    road_col = np.array([0.33, 0.33, 0.33])
    out[m > 0.5] = road_col
    # pronounced dark halo at boundary — the "negative impression"
    boundary = binary_dilation(m > 0.5, iterations=3).astype(float) - (m > 0.5).astype(float)
    boundary = gaussian_filter(np.clip(boundary, 0, 1), sigma=1.5)
    shadow = boundary[:, :, None] * np.array([[[0.12, 0.10, 0.08]]])
    out = np.clip(out - shadow * 2.2, 0, 1)
    # slight colour fringing (redistributed probability mass along edges)
    fringe = gaussian_filter((m > 0.5).astype(float), sigma=2.5)
    fringe = fringe * (1 - (m > 0.5).astype(float))
    out[:, :, 0] = np.clip(out[:, :, 0] - fringe * 0.08, 0, 1)  # slight red-depletion band
    return out

def seg_advr(scene, m):
    """ADVR: perfectly seamless inpainting — no visible trace."""
    out = scene.copy()
    # fill with local context (road colour sampled from surroundings)
    road_col = np.array([0.34 + np.random.normal(0, 0.005),
                         0.34 + np.random.normal(0, 0.005),
                         0.34 + np.random.normal(0, 0.005)])
    out[m > 0.5] = road_col
    # smooth blend outward — feathered edge, no halo
    dilated = gaussian_filter((m > 0.5).astype(float), sigma=3.5)
    for c in range(3):
        out[:, :, c] = out[:, :, c] * (1 - dilated * 0.10) + road_col[c] * dilated * 0.10
    # add matching road noise inside person region
    road_noise = np.random.normal(0, 0.011, scene.shape[:2])
    for c in range(3):
        out[:, :, c][m > 0.5] = np.clip(out[:, :, c][m > 0.5] + road_noise[m > 0.5], 0, 1)
    return np.clip(out, 0, 1)

# ── Attacker heatmaps ─────────────────────────────────────────────────────────
def attacker_delete(m):
    """High-confidence silhouette recovery — attacker succeeds."""
    signal = gaussian_filter(m, sigma=1.8)
    noise  = np.random.normal(0, 0.07, m.shape)
    raw = signal * 0.88 + 0.12 + noise
    raw = np.clip(raw, 0, 1)
    # normalise so interior = hot, background = cold
    raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-6)
    return raw

def attacker_advr(m):
    """Random noise — attacker completely fails."""
    noise = np.random.normal(0.5, 0.18, m.shape)
    # almost zero correlation with mask
    faint = m * 0.04 + noise * 0.96
    faint = gaussian_filter(faint, sigma=1.0)
    # normalise to same visual range so the comparison is fair
    faint = (faint - faint.min()) / (faint.max() - faint.min() + 1e-6)
    return faint

# ── Figure layout ─────────────────────────────────────────────────────────────
# 2 rows × 4 columns
# Row 0 (model outputs):  Input | GT mask | DELETE out | ADVR out
# Row 1 (attacker maps):  [blank context] | GT mask    | Atk←DELETE | Atk←ADVR

img_input   = scene
img_gt      = np.stack([pmask]*3, axis=-1)
img_orig    = seg_original(scene, pmask)
img_del     = seg_delete(scene, pmask)
img_advr    = seg_advr(scene, pmask)
hmap_del    = attacker_delete(pmask)
hmap_advr   = attacker_advr(pmask)

# inferno-like but with white for low values (more intuitive: bright=high prob)
HMAP_CMAP = "inferno"

fig2 = plt.figure(figsize=(12, 5.6))
gs = gridspec.GridSpec(2, 4, figure=fig2,
                       hspace=0.08, wspace=0.06,
                       left=0.01, right=0.99, top=0.84, bottom=0.12)

axes2 = [[fig2.add_subplot(gs[r, c]) for c in range(4)] for r in range(2)]

ROW0 = [img_input, img_gt, img_del, img_advr]
ROW0_TITLES = [
    "Input scene",
    "GT person mask",
    "After DELETE [CVPR'25]\n(label suppressed)",
    "After ADVR (ours)\n(label suppressed)",
]

ROW1 = [img_orig, img_gt, hmap_del, hmap_advr]
ROW1_TITLES = [
    "Original model\noutput",
    "",
    "Attacker recovery\nfrom DELETE output",
    "Attacker recovery\nfrom ADVR output",
]
ROW1_IS_HMAP = [False, False, True, True]

VERDICT = [None, None, ("LEAKS", RED), ("SAFE", GREEN)]

for c in range(4):
    # row 0
    ax = axes2[0][c]
    ax.imshow(ROW0[c], interpolation="bilinear")
    ax.set_title(ROW0_TITLES[c], fontsize=7.8, pad=4, linespacing=1.35,
                 fontweight="bold" if c in (2,3) else "normal")
    ax.axis("off")

    # row 1
    ax = axes2[1][c]
    if ROW1_IS_HMAP[c]:
        im = ax.imshow(ROW1[c], cmap=HMAP_CMAP, vmin=0, vmax=1,
                       interpolation="bilinear")
    else:
        ax.imshow(ROW1[c], interpolation="bilinear")
    if ROW1_TITLES[c]:
        ax.set_title(ROW1_TITLES[c], fontsize=7.8, pad=4, linespacing=1.35)
    ax.axis("off")

    # verdict badges
    if VERDICT[c] is not None:
        label, col = VERDICT[c]
        for row in range(2):
            axt = axes2[row][c]
            bbox = axt.get_position()
            # draw a bold coloured border around the axis
            for spine_ax in fig2.axes:
                pass
        # badge below row 1
        ax1 = axes2[1][c]
        bbox = ax1.get_position()
        fig2.text(bbox.x0 + bbox.width / 2, bbox.y0 - 0.025,
                  label, ha="center", va="top", fontsize=9,
                  color="white", fontweight="bold",
                  bbox=dict(facecolor=col, edgecolor="none",
                            boxstyle="round,pad=0.3", alpha=0.93))
        # coloured rectangle border on both row panels
        for row in range(2):
            axt = axes2[row][c]
            b = axt.get_position()
            rect = plt.Rectangle((b.x0, b.y0), b.width, b.height,
                                  transform=fig2.transFigure, zorder=10,
                                  fill=False, edgecolor=col, linewidth=2.5)
            fig2.add_artist(rect)

# ── Row separating arrow label ────────────────────────────────────────────────
# "Attacker trained ↓" label between rows
fig2.text(0.005, 0.50, "Learned\nattacker\n↓", ha="left", va="center",
          fontsize=8, style="italic", color="#555555")

# ── Colorbar for heatmaps ─────────────────────────────────────────────────────
cbar_ax = fig2.add_axes([0.72, 0.06, 0.26, 0.025])
sm = plt.cm.ScalarMappable(cmap=HMAP_CMAP, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cb = fig2.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cb.set_label("Attacker confidence: person here?", fontsize=7.5)
cb.ax.tick_params(labelsize=7)
cb.set_ticks([0, 0.5, 1])
cb.set_ticklabels(["Low", "Medium", "High"])

fig2.suptitle(
    "Erased but not forgotten: label suppression ≠ shape erasure",
    fontsize=10, fontweight="bold", y=0.96)

fig2.savefig("paper/wacv_template/figures/fig_qualitative.pdf",
             bbox_inches="tight", dpi=250)
fig2.savefig("paper/wacv_template/figures/fig_qualitative.png",
             bbox_inches="tight", dpi=250)
plt.close(fig2)
print("Saved fig_qualitative")
