"""ADE20K semantic-segmentation data for class-unlearning.

Conventions (mmseg-style):
  * 150 valid classes, label index 0..149 (raw PNG value minus 1).
  * raw value 0 -> IGNORE (255).
  * PERSON is raw class 13 -> index 12.

The unlearning target is built by **dominant-neighbor relabeling**: every person
pixel is replaced by the label of the nearest non-person pixel (a distance-
transform fill of the label map). This is what defines both the retrain oracle
and the "did it blend in?" leakage metric -- a forgotten region should look like
its surroundings, not leave a person-shaped hole.
"""
import glob
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

# Repo root = the parent of this file's package dir (project1/seg/data.py -> project1/).
_REPO_ROOT = Path(__file__).resolve().parent.parent

# ADE20K isn't redistributed in this repo (licensing) — download it yourself (see
# docs/REPRODUCING.md) and either symlink it to data/ADE20K or point ADE20K_ROOT at
# your copy. Expected layout: <root>/images/{training,validation}_raw/*.jpg,
# <root>/annotations/{training,validation}/*.png (mmseg-style extraction).
ADE_ROOT = os.environ.get("ADE20K_ROOT", str(_REPO_ROOT / "data" / "ADE20K"))
# Dataset-dependent globals, env-configurable so one codebase serves both
# ADE20K (150 classes, person idx 12) and Cityscapes (19 classes, person idx 11).
#   FORGET_CLS  -> index of the class to unlearn (ADE20K person=12, car=20;
#                  Cityscapes person=11, rider=12)
#   NUM_CLASSES -> 150 (ADE20K, default) or 19 (Cityscapes)
import os as _os
PERSON = int(_os.environ.get("FORGET_CLS", "12"))
IGNORE = 255
NUM_CLASSES = int(_os.environ.get("NUM_CLASSES", "150"))

# Cityscapes: download from cityscapes-dataset.com and extract to data/cityscapes
# (or point CITYSCAPES_ROOT at your copy). See docs/REPRODUCING.md.
CITY_ROOT = os.environ.get("CITYSCAPES_ROOT", str(_REPO_ROOT / "data" / "cityscapes"))
# Cityscapes labelId (0..33) -> trainId (0..18), else IGNORE. Standard 19-class map.
_CITY_LABELID_TO_TRAINID = np.full(34, IGNORE, dtype=np.int64)
for _lid, _tid in {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
                   22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16,
                   32: 17, 33: 18}.items():
    _CITY_LABELID_TO_TRAINID[_lid] = _tid

_IMNET_MEAN = [0.485, 0.456, 0.406]
_IMNET_STD = [0.229, 0.224, 0.225]


# RELABEL selects how person pixels are relabelled when building the oracle target.
# "nn" (default) = dominant-neighbour nearest-valid fill (the paper's scheme).
# The T2.1 ablation retrains the oracle under each scheme to show the leakage FLOOR
# does not depend on this preprocessing choice.
RELABEL = _os.environ.get("RELABEL", "nn")


def merge_person_label(label, person_idx=None, ignore=IGNORE, scheme=None):
    """Replace person pixels with a valid (non-person, non-ignore) label so a
    forgotten region blends into its surroundings rather than leaving a hole.
    `label` is an int array with values 0..NUM_CLASSES-1 or `ignore`.

    scheme (defaults to env RELABEL):
      "nn"     nearest valid donor pixel (distance-transform fill) -- the default.
      "mode"   the single most frequent valid class in the image (global dominant).
      "rand"   each person pixel gets a random valid donor label from the image.
      "ignore" person pixels set to `ignore` (unsupervised in the oracle).
    """
    if person_idx is None:
        person_idx = PERSON
    if scheme is None:
        scheme = RELABEL
    mask = label == person_idx
    if not mask.any():
        return label
    valid = (label != person_idx) & (label != ignore)
    if not valid.any():
        return label  # nothing real to blend into; leave as-is
    out = label.copy()
    if scheme == "ignore":
        out[mask] = ignore
        return out
    if scheme == "nn":
        inds = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        out[mask] = label[tuple(inds)][mask]
        return out
    donors = label[valid]
    if scheme == "mode":
        vals, cnts = np.unique(donors, return_counts=True)
        out[mask] = vals[cnts.argmax()]
        return out
    if scheme == "rand":
        out[mask] = np.random.choice(donors, size=int(mask.sum()), replace=True)
        return out
    raise ValueError(f"unknown relabel scheme {scheme!r}")


class ADE20KSeg(Dataset):
    """Returns (image[3,H,W] float, label[H,W] long).

    target="orig"   -> original labels (train the original model)
    target="merged" -> person dominant-neighbor relabeled (train the oracle)
    person_only=True -> restrict to images that contain a person (fast iteration)
    """

    def __init__(self, split="train", target="orig", size=512, crop=True,
                 person_only=False, root=ADE_ROOT):
        sub = "training" if split == "train" else "validation"
        self.img_dir = os.path.join(root, "images", f"{sub}_raw")
        self.ann_dir = os.path.join(root, "annotations", sub)
        self.anns = sorted(glob.glob(os.path.join(self.ann_dir, "*.png")))
        self.split = split
        self.target = target
        self.size = size
        self.crop = crop and split == "train"
        if person_only:
            self.anns = [a for a in self.anns if self._has_person(a)]

    @staticmethod
    def _has_person(ann_path):
        return (np.array(Image.open(ann_path)) == PERSON + 1).any()

    def __len__(self):
        return len(self.anns)

    def _load(self, idx):
        ann_path = self.anns[idx]
        stem = os.path.splitext(os.path.basename(ann_path))[0]
        img = Image.open(os.path.join(self.img_dir, stem + ".jpg")).convert("RGB")
        raw = np.array(Image.open(ann_path)).astype(np.int64)
        # reduce_zero_label: 0 -> ignore, else -1
        label = raw - 1
        label[raw == 0] = IGNORE
        return img, label

    def __getitem__(self, idx):
        img, label = self._load(idx)
        if self.target == "merged":
            label = merge_person_label(label)
        label = Image.fromarray(label.astype(np.int32), mode="I")

        # geometry
        if self.crop:                       # train: random scale (0.5-2.0) + crop + flip
            s = self.size
            scale = 0.5 + 1.5 * torch.rand(1).item()
            new = max(int(round(s * scale)), s)
            img = TF.resize(img, [new, new], TF.InterpolationMode.BILINEAR)
            label = TF.resize(label, [new, new], TF.InterpolationMode.NEAREST)
            top = int(torch.randint(0, new - s + 1, (1,)).item())
            left = int(torch.randint(0, new - s + 1, (1,)).item())
            img = TF.crop(img, top, left, s, s)
            label = TF.crop(label, top, left, s, s)
            if torch.rand(1).item() < 0.5:
                img = TF.hflip(img); label = TF.hflip(label)
        else:                               # eval: plain resize
            img = TF.resize(img, [self.size, self.size], TF.InterpolationMode.BILINEAR)
            label = TF.resize(label, [self.size, self.size], TF.InterpolationMode.NEAREST)

        img = TF.to_tensor(img)
        img = TF.normalize(img, _IMNET_MEAN, _IMNET_STD)
        label = torch.from_numpy(np.array(label)).long()
        return img, label


class ADE20KUnlearn(Dataset):
    """Returns (image, orig_label, merged_label) for the unlearning methods.

    Restricted to person-containing images (the forget signal). `orig_label` keeps
    person; `merged_label` has person dominant-neighbor relabeled. The person mask
    is `orig_label == PERSON`; retain pixels are everything else.
    """

    def __init__(self, split="train", size=512, crop=True, root=ADE_ROOT):
        self.base = ADE20KSeg(split, target="orig", size=size, crop=crop,
                              person_only=True, root=root)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # reuse the base loader for geometry-consistent image + orig label, then
        # derive the merged label from the same (post-transform) orig label.
        img, orig = self.base[idx]
        merged = torch.from_numpy(
            merge_person_label(orig.numpy().copy())).long()
        return img, orig, merged


# --------------------------------------------------------------------------- #
# Cityscapes (19-class street scenes). person = trainId 11 (default forget class
# for Cityscapes runs, set FORGET_CLS=11 + NUM_CLASSES=19).
# --------------------------------------------------------------------------- #
def _geom(img, label, size, crop):
    """Shared geometry: train = random scale+crop+flip; eval = resize. `label` is a
    PIL 'I' image; returns (img_tensor, label_tensor)."""
    if crop:
        s = size
        scale = 0.5 + 1.5 * torch.rand(1).item()
        new = max(int(round(s * scale)), s)
        img = TF.resize(img, [new, new], TF.InterpolationMode.BILINEAR)
        label = TF.resize(label, [new, new], TF.InterpolationMode.NEAREST)
        top = int(torch.randint(0, new - s + 1, (1,)).item())
        left = int(torch.randint(0, new - s + 1, (1,)).item())
        img = TF.crop(img, top, left, s, s); label = TF.crop(label, top, left, s, s)
        if torch.rand(1).item() < 0.5:
            img = TF.hflip(img); label = TF.hflip(label)
    else:
        img = TF.resize(img, [size, size], TF.InterpolationMode.BILINEAR)
        label = TF.resize(label, [size, size], TF.InterpolationMode.NEAREST)
    img = TF.normalize(TF.to_tensor(img), _IMNET_MEAN, _IMNET_STD)
    return img, torch.from_numpy(np.array(label)).long()


class CityscapesSeg(Dataset):
    """Cityscapes semantic segmentation, 19 trainId classes. Mirrors ADE20KSeg."""

    def __init__(self, split="train", target="orig", size=512, crop=True,
                 person_only=False, root=CITY_ROOT):
        sub = "val" if split == "val" else split          # train/val/test
        self.img_dir = os.path.join(root, "leftImg8bit", sub)
        self.ann_dir = os.path.join(root, "gtFine", sub)
        self.anns = sorted(glob.glob(os.path.join(self.ann_dir, "*", "*_labelIds.png")))
        self.target = target; self.size = size
        self.crop = crop and split == "train"
        if person_only:
            self.anns = [a for a in self.anns if self._has_person(a)]

    @staticmethod
    def _has_person(ann_path):
        return (_CITY_LABELID_TO_TRAINID[np.array(Image.open(ann_path))] == PERSON).any()

    def __len__(self):
        return len(self.anns)

    def _load(self, idx):
        ann_path = self.anns[idx]
        base = os.path.basename(ann_path).replace("_gtFine_labelIds.png", "")
        city = os.path.basename(os.path.dirname(ann_path))
        img = Image.open(os.path.join(self.img_dir, city,
                                      base + "_leftImg8bit.png")).convert("RGB")
        label = _CITY_LABELID_TO_TRAINID[np.array(Image.open(ann_path))].copy()
        return img, label

    def __getitem__(self, idx):
        img, label = self._load(idx)
        if self.target == "merged":
            label = merge_person_label(label)
        label = Image.fromarray(label.astype(np.int32), mode="I")
        return _geom(img, label, self.size, self.crop)


class CityscapesUnlearn(Dataset):
    """(image, orig_label, merged_label) over person-containing Cityscapes images."""

    def __init__(self, split="train", size=512, crop=True, root=CITY_ROOT):
        self.base = CityscapesSeg(split, target="orig", size=size, crop=crop,
                                  person_only=True, root=root)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, orig = self.base[idx]
        merged = torch.from_numpy(merge_person_label(orig.numpy().copy())).long()
        return img, orig, merged


# Dataset selector: scripts import SegDS / UnlearnDS and get the right classes.
DATASET = _os.environ.get("DATASET", "ade20k")
if DATASET == "cityscapes":
    SegDS, UnlearnDS = CityscapesSeg, CityscapesUnlearn
else:
    SegDS, UnlearnDS = ADE20KSeg, ADE20KUnlearn
