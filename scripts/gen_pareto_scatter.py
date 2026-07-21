"""
Pareto scatter: retain_mIoU (x) vs leakage AUC (y).
One point per method. Shows the three failure modes as distinct quadrants.
Data from Tables 1+2 in the paper (ADE20K, seed 1).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch

OUT_DIR = "paper/wacv_template/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data from paper tables (ADE20K, DeepLabV3-R50) ──────────────────────────
# (name, retain_mIoU, out_AUC, family)
# family: "destroy" | "leak" | "safe" | "oracle" | "ours"
METHODS = [
    # Destroyed methods
    ("NegGrad",   0.1,  50.0,  "destroy"),   # NaN → 50 (random = chance)
    ("SCRUB",     0.1,  50.0,  "destroy"),
    ("SSD",       0.1,  66.1,  "destroy"),
    ("B1",        0.4,  77.1,  "destroy"),

    # Silent leakage baselines
    ("FT",        9.3,  90.4,  "leak"),
    ("NegGrad+",  7.7,  91.0,  "leak"),
    ("RL",        5.2,  96.9,  "leak"),
    ("SalUn",     8.0,  97.2,  "leak"),
    ("DELETE",   12.4,  93.7,  "leak"),
    ("DELETE\n(logitmask)", 36.6, 96.4, "leak"),

    # Our method + ablations
    ("A2",       23.7,  84.7,  "ours_abl"),
    ("ADV",      21.4,  82.0,  "ours_abl"),
    ("ADVR\n(ours)", 26.8, 81.6, "ours"),

    # Gold standard
    ("Oracle",   36.6,  84.8,  "oracle"),
]

ORACLE_FLOOR = 84.8
FLOOR_STD    = 0.5
THRESHOLD    = ORACLE_FLOOR + 2 * FLOOR_STD   # 85.8

STYLE = {
    "destroy":  dict(color="#888888", marker="X",  s=140, zorder=4, label="Destroyed"),
    "leak":     dict(color="#d62728", marker="o",  s=140, zorder=4, label="Silhouette leakage"),
    "ours_abl": dict(color="#1f77b4", marker="D",  s=100, zorder=4, label="Ours (ablation)"),
    "ours":     dict(color="#2ca02c", marker="*",  s=380, zorder=6, label="ADVR (ours)"),
    "oracle":   dict(color="#ff7f0e", marker="s",  s=160, zorder=5, label="Oracle (gold std)"),
}

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 5.4))
fig.patch.set_facecolor("white")
ax.set_facecolor("#fafafa")

# Background quadrant shading
ax.axhspan(THRESHOLD, 102, color="#ffeeee", alpha=0.55, zorder=0)   # leaks zone
ax.axhspan(40,  THRESHOLD, color="#eeffee", alpha=0.35, zorder=0)   # safe zone

# Oracle floor band
ax.axhline(ORACLE_FLOOR, color="#ff7f0e", ls="--", lw=1.5, alpha=0.8, zorder=3,
           label=f"Oracle floor ({ORACLE_FLOOR:.1f})")
ax.axhline(THRESHOLD, color="#d62728", ls=":", lw=1.5, alpha=0.8, zorder=3,
           label=f"2σ leak threshold ({THRESHOLD:.1f})")

# Quadrant labels
ax.text(37.5, 99.5, "LEAKS", fontsize=9, color="#c0392b", fontweight="bold",
        ha="right", va="top", alpha=0.7)
ax.text(37.5, THRESHOLD - 0.8, "SAFE", fontsize=9, color="#27ae60", fontweight="bold",
        ha="right", va="top", alpha=0.8)
ax.text(1.5, 99.5, "DESTROYED\n(model collapses)", fontsize=7.5, color="#666666",
        ha="left", va="top", alpha=0.7, linespacing=1.4)

# Plot points
seen = set()
for name, retain, auc, fam in METHODS:
    kw = STYLE[fam]
    lbl = kw["label"] if fam not in seen else None
    seen.add(fam)
    ax.scatter(retain, auc, color=kw["color"], marker=kw["marker"],
               s=kw["s"], zorder=kw["zorder"], label=lbl,
               edgecolors="white" if fam != "destroy" else "#555555", linewidths=0.8)

# Annotate key points
ANNOT = {
    "ADVR\n(ours)":         (1.5,  -1.2,  "#2ca02c"),
    "Oracle":               (1.5,  -1.2,  "#ff7f0e"),
    "DELETE":               (1.2,   1.0,  "#d62728"),
    "DELETE\n(logitmask)":  (-2.5, -2.5,  "#d62728"),
    "SalUn":                (0.8,   1.0,  "#d62728"),
    "FT":                   (0.8,  -2.2,  "#d62728"),
    "SSD":                  (1.0,   1.2,  "#888888"),
    "A2":                   (0.8,  -2.2,  "#1f77b4"),
    "ADV":                  (0.8,   1.0,  "#1f77b4"),
}
for name, retain, auc, fam in METHODS:
    display = name.replace("\n", " ")
    if name in ANNOT:
        dx, dy, col = ANNOT[name]
        ax.annotate(display,
                    xy=(retain, auc), xytext=(retain + dx, auc + dy),
                    fontsize=7.5, color=col, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.6, alpha=0.5))

# Arrow showing "ideal direction"
ax.annotate("", xy=(38, 80), xytext=(32, 80),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))
ax.text(35, 79.2, "↑ higher retain", fontsize=7, color="#555555", ha="center")
ax.annotate("", xy=(38, 80), xytext=(38, 85),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))
ax.text(38.8, 82.5, "↓ less\nleakage", fontsize=7, color="#555555",
        ha="left", va="center")

ax.set_xlabel("Retain mIoU ↑  (non-person classes)", fontsize=11)
ax.set_ylabel("Attacker AUC  (leakage ↑ = worse)", fontsize=11)
ax.set_title("Pareto Analysis: Retain Utility vs. Spatial Privacy",
             fontsize=12, fontweight="bold", pad=10)

ax.set_xlim(-1, 41)
ax.set_ylim(44, 101)
ax.tick_params(labelsize=9)
ax.grid(True, alpha=0.25, lw=0.6)

legend = ax.legend(fontsize=8.5, framealpha=0.9, loc="lower right",
                   edgecolor="#cccccc", handletextpad=0.5)

# Star annotation for ADVR
ax.annotate("Only method with\nhigh retain AND\nno leakage",
            xy=(26.8, 81.6), xytext=(18, 74),
            fontsize=8, color="#2ca02c", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.3),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ca02c", lw=1.2))

fig.tight_layout()
for ext in ("pdf", "png"):
    p = f"{OUT_DIR}/fig_pareto.{ext}"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved → {p}")
plt.close(fig)
