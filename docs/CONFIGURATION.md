# Configuration reference

There's no YAML/JSON config system — experiment parameters are set either via CLI
flags (per-script, see `--help` / each script's docstring) or via environment
variables. This lets one codebase serve every dataset/class/backbone axis without
code changes. The core ones are collected in [`configs/axes.sh`](../configs/axes.sh);
this page documents what each variable actually does.

## Core axis variables

| Variable | Default | Meaning |
|---|---|---|
| `DATASET` | `ade20k` | `ade20k` or `cityscapes` — selects `seg.data.SegDS`/`UnlearnDS`. |
| `FORGET_CLS` | `12` | Class index to unlearn. ADE20K: person=12, car=20. Cityscapes: person=11, rider=12. |
| `NUM_CLASSES` | `150` | `150` for ADE20K, `19` for Cityscapes. |
| `ARCH` | `deeplabv3r50` | `deeplabv3r50` or `segformer_b2` (see `seg/model.py`). SegFormer needs `pip install -e '.[segformer]'`. |
| `RELABEL` | `nn` | How the oracle's person-pixel relabeling target is built: `nn` (nearest-neighbor fill, the paper default), `mode`, `rand`, `ignore`. See `seg/data.py::merge_person_label`. |

## Dataset locations

| Variable | Default | Meaning |
|---|---|---|
| `ADE20K_ROOT` | `<repo>/data/ADE20K` | Root of an mmseg-style-extracted ADE20K (`images/{training,validation}_raw/*.jpg`, `annotations/{training,validation}/*.png`). Not redistributed here — download and symlink/set this. |
| `CITYSCAPES_ROOT` | `<repo>/data/cityscapes` | Root of a Cityscapes extraction (`leftImg8bit/`, `gtFine/`). |

## Leakage-attack protocol (`scripts/run_attack_variance.py` and friends)

| Variable | Default | Meaning |
|---|---|---|
| `UNLEARNED` | `ADV,ADVR,ADVR_s1,B1` | Comma-separated `checkpoints/unlearned_<name>.pth` basenames to attack, in addition to the oracle floor and DELETE-logitmask prior-art (always included). |
| `N_TRIALS` | `3` | Attacker retrainings per model/surface (mean±std). Bump to `10` for publication-grade CIs (reviewer W2). |
| `OUT_JSON` | `results/seg/attack_variance.json` | Where the result table is written. |

## Miscellaneous

| Variable | Default | Meaning |
|---|---|---|
| `PYTORCH_ALLOC_CONF` | unset | Set to `expandable_segments:True` in the provided scripts to reduce CUDA OOM from fragmentation on long unlearning runs. |

## Rebuttal-experiment-specific variables

`scripts/experiments/rebuttal/*.py` each take a handful of additional env vars
specific to that experiment (attacker architecture/capacity, restart counts, etc.)
— documented in each script's module docstring rather than duplicated here; see
[`docs/REVIEWER_EXPERIMENTS.md`](REVIEWER_EXPERIMENTS.md) for the full map from
reviewer concern → script → output file, and `SMOKE=1` on
`exp_reviewer_response.sh` to dry-run the whole queue in minutes.
