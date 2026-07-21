"""Candidate unlearning methods from METHOD_IDEAS.md, tested one by one.

Shared design: a frozen copy of the original anchors retain pixels (KL), so updates
stay localized (fixes the GA/SSD destruction). The person region is handled
differently per idea (label CE / feature collapse / orthogonal ascent).
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from seg.data import IGNORE, NUM_CLASSES, PERSON
from seg.model import seg_forward
from seg.utils import clone_model as _clone
from seg.utils import masked_ce as _masked_ce
from seg.utils import pixel_masks as _masks


def _frozen(model, device):
    t = _clone(model).to(device).eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


def _freeze_bn(model):
    """Put all BatchNorm layers in eval mode so their running stats don't drift
    toward the (biased) person-image subset during unlearning."""
    for mod in model.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.eval()


def _masked_kl(z_s, z_t, mask, T=1.0):
    """KL(student || teacher) averaged over masked pixels (anchor retain)."""
    if mask.sum() == 0:
        return z_s.sum() * 0.0
    ls = F.log_softmax(z_s / T, 1)
    pt = F.softmax(z_t / T, 1)
    kld = F.kl_div(ls, pt, reduction="none").sum(1)        # [N,H,W]
    return (kld * mask).sum() / mask.sum().clamp(min=1) * (T * T)


# --------------------------------------------------------------------------- #
# A2 — retain-anchored finetune-to-merged
# --------------------------------------------------------------------------- #
def ft_anchored(model, loader, device="cuda", epochs=3, lr=1e-3, lam_r=1.0,
                freeze_bn=False, **_):
    m = _clone(model).to(device).train()
    if freeze_bn:
        _freeze_bn(m)
    teacher = _frozen(model, device)
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    for _e in range(epochs):
        if freeze_bn:
            _freeze_bn(m)
        for img, orig, merged in loader:
            img, orig, merged = img.to(device), orig.to(device), merged.to(device)
            forget, retain = _masks(orig)
            z = seg_forward(m, img)
            with torch.no_grad():
                z0 = seg_forward(teacher, img)
            loss = _masked_ce(z, merged, forget) + lam_r * _masked_kl(z, z0, retain)
            opt.zero_grad(); loss.backward(); opt.step()
    return m


# --------------------------------------------------------------------------- #
# D1 — retain-orthogonal gradient ascent (+ optional blend)
# --------------------------------------------------------------------------- #
def ga_retain_orth(model, loader, device="cuda", epochs=2, lr=1e-4, blend=False, **_):
    m = _clone(model).to(device).train()
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)
    params = [p for p in m.parameters() if p.requires_grad]
    for _e in range(epochs):
        for img, orig, merged in loader:
            img, orig = img.to(device), orig.to(device)
            forget, retain = _masks(orig)
            z = seg_forward(m, img)
            # retain grad (preserve)
            opt.zero_grad()
            _masked_ce(z, orig, retain).backward(retain_graph=True)
            gR = [p.grad.detach().clone() if p.grad is not None else None for p in params]
            # forget grad (ascent target)
            opt.zero_grad()
            tgt = merged.to(device) if blend else orig
            f_loss = -_masked_ce(z, orig, forget) if not blend else _masked_ce(z, tgt, forget)
            f_loss.backward()
            gF = [p.grad.detach().clone() if p.grad is not None else None for p in params]
            # project gF orthogonal to gR
            dot = sum((f * r).sum() for f, r in zip(gF, gR) if f is not None and r is not None)
            nrm = sum((r * r).sum() for r in gR if r is not None).clamp(min=1e-12)
            coef = (dot / nrm)
            opt.zero_grad()
            for p, f, r in zip(params, gF, gR):
                if f is None:
                    continue
                p.grad = f - coef * r if r is not None else f
            opt.step()
    return m


# --------------------------------------------------------------------------- #
# B1 — neighbor-prototype feature collapse
# --------------------------------------------------------------------------- #
class _FeatHook:
    """Capture the 256-d input to DeepLabV3's final 1x1 classifier conv."""
    def __init__(self, model):
        self.feat = None
        self.h = model.classifier[4].register_forward_pre_hook(self._hook)

    def _hook(self, module, inp):
        self.feat = inp[0]

    def close(self):
        self.h.remove()


@torch.no_grad()
def _class_prototypes(teacher, loader, device, n_batches=60):
    hook = _FeatHook(teacher)
    D = teacher.classifier[4].in_channels
    sums = torch.zeros(NUM_CLASSES, D, device=device)
    cnts = torch.zeros(NUM_CLASSES, device=device)
    for i, (img, orig, _merged) in enumerate(loader):
        if i >= n_batches:
            break
        seg_forward(teacher, img.to(device))
        feat = hook.feat                                   # [N,D,h,w]
        N, Dd, h, w = feat.shape
        y = F.interpolate(orig.float().unsqueeze(1).to(device), size=(h, w),
                          mode="nearest").long().squeeze(1)   # [N,h,w]
        feat = feat.permute(0, 2, 3, 1).reshape(-1, Dd)        # [Nhw,D]
        yl = y.reshape(-1)
        valid = (yl != IGNORE) & (yl != PERSON)
        sums.index_add_(0, yl[valid], feat[valid])
        cnts.index_add_(0, yl[valid], torch.ones(valid.sum(), device=device))
    hook.close()
    proto = sums / cnts.clamp(min=1).unsqueeze(1)
    return proto                                            # [C,D]


def proto_collapse(model, loader, device="cuda", epochs=3, lr=1e-3, lam_r=1.0,
                   lam_c=1.0, lam_f=0.0, freeze_bn=False, **_):
    """B1/F1: collapse person features onto the neighbor prototype (robustness),
    anchor retain in logit space (lam_r) and optionally feature space (lam_f),
    optionally freeze BN to kill the forget-subset distribution shift."""
    m = _clone(model).to(device).train()
    if freeze_bn:
        _freeze_bn(m)
    teacher = _frozen(model, device)
    proto = _class_prototypes(teacher, loader, device)
    hook = _FeatHook(m)
    hook_t = _FeatHook(teacher)
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    for _e in range(epochs):
        if freeze_bn:
            _freeze_bn(m)
        for img, orig, merged in loader:
            img, orig, merged = img.to(device), orig.to(device), merged.to(device)
            z = seg_forward(m, img)                         # sets hook.feat
            feat = hook.feat                                # [N,D,h,w]
            N, D, h, w = feat.shape
            yl = F.interpolate(orig.float().unsqueeze(1), size=(h, w),
                               mode="nearest").long().squeeze(1)
            ml = F.interpolate(merged.float().unsqueeze(1), size=(h, w),
                               mode="nearest").long().squeeze(1)
            fmask = (yl == PERSON)
            rmask = (yl != PERSON) & (yl != IGNORE)
            feat_p = feat.permute(0, 2, 3, 1)               # [N,h,w,D]
            with torch.no_grad():
                z0 = seg_forward(teacher, img)
                feat0 = hook_t.feat.permute(0, 2, 3, 1)     # [N,h,w,D]
            collapse = (((feat_p[fmask] - proto[ml[fmask].clamp(0, NUM_CLASSES - 1)]) ** 2)
                        .sum(1).mean() if fmask.any() else feat.sum() * 0.0)
            feat_anchor = (((feat_p[rmask] - feat0[rmask]) ** 2).sum(1).mean()
                           if (lam_f > 0 and rmask.any()) else feat.sum() * 0.0)
            _, retain = _masks(orig)
            loss = lam_c * collapse + lam_r * _masked_kl(z, z0, retain) \
                + lam_f * feat_anchor
            opt.zero_grad(); loss.backward(); opt.step()
    hook.close(); hook_t.close()
    return m


# --------------------------------------------------------------------------- #
# ADV — adversarial silhouette removal (the HYP method)
# --------------------------------------------------------------------------- #
def adv_silhouette(model, loader, device="cuda", epochs=3, lr=1e-3, lam_r=1.0,
                   lam_adv=1.0, lr_d=1e-3, out_size=128, freeze_bn=False, **_):
    """min_G [ blend(person->dominant-neighbor) + lam_r*retain-anchor
              + lam_adv*fool(D) ]   ;   max_D  BCE(D(O), M).
    D is the silhouette discriminator on the output field; fooling it drives the
    spatial person-information I(M;O) -> 0 while the blend+anchor keep utility."""
    from seg.attack import SilhouetteNet, person_mask_low

    m = _clone(model).to(device).train()
    if freeze_bn:
        _freeze_bn(m)
    teacher = _frozen(model, device)
    D = SilhouetteNet().to(device).train()
    optG = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    optD = torch.optim.Adam(D.parameters(), lr=lr_d)
    bce = nn.BCEWithLogitsLoss()

    def field(z):
        return F.interpolate(F.softmax(z, 1), size=(out_size, out_size),
                             mode="bilinear", align_corners=False)

    for _e in range(epochs):
        if freeze_bn:
            _freeze_bn(m)
        for img, orig, merged in loader:
            img, orig, merged = img.to(device), orig.to(device), merged.to(device)
            forget, retain = _masks(orig)
            Mlow = person_mask_low(orig, out_size)               # [N,h,w] in {0,1}

            # ---- D step: learn to reconstruct the silhouette ----
            with torch.no_grad():
                p_det = field(seg_forward(m, img))
            optD.zero_grad(); bce(D(p_det), Mlow).backward(); optD.step()

            # ---- G step: blend + anchor + fool D ----
            z = seg_forward(m, img)
            with torch.no_grad():
                z0 = seg_forward(teacher, img)
            loss = _masked_ce(z, merged, forget) + lam_r * _masked_kl(z, z0, retain)
            dpred = D(field(z))
            fmask = Mlow > 0.5
            if fmask.any():                                      # make person region look like context
                loss = loss + lam_adv * bce(dpred[fmask], torch.zeros_like(dpred[fmask]))
            optG.zero_grad(); loss.backward(); optG.step()
    return m


# --------------------------------------------------------------------------- #
# ADV-FULL — defend BOTH surfaces + retain replay (the strengthened method)
# --------------------------------------------------------------------------- #
def adv_full(model, loader, device="cuda", epochs=6, lr=1e-3, lam_r=1.0, lam_adv=1.0,
             lam_c=1.0, lr_d=1e-3, out_size=128, replay_loader=None, **_):
    """Output-adversarial (fool silhouette D) + feature collapse to neighbor
    prototype (kill the feature attack) + retain KL anchor + retain replay on the
    full image distribution (close the retain gap)."""
    from seg.attack import SilhouetteNet, person_mask_low

    m = _clone(model).to(device).train()
    teacher = _frozen(model, device)
    use_collapse = lam_c > 0
    proto = _class_prototypes(teacher, loader, device) if use_collapse else None
    hook = _FeatHook(m) if use_collapse else None      # skip when unused -> less memory
    D = SilhouetteNet().to(device).train()
    optG = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    optD = torch.optim.Adam(D.parameters(), lr=lr_d)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE)
    replay_it = iter(replay_loader) if replay_loader is not None else None

    def field(z):
        return F.interpolate(F.softmax(z, 1), size=(out_size, out_size),
                             mode="bilinear", align_corners=False)

    for _e in range(epochs):
        for img, orig, merged in loader:
            img, orig, merged = img.to(device), orig.to(device), merged.to(device)
            forget, retain = _masks(orig)
            Mlow = person_mask_low(orig, out_size)

            # ---- D step ----
            with torch.no_grad():
                p_det = field(seg_forward(m, img))
            optD.zero_grad(); bce(D(p_det), Mlow).backward(); optD.step()

            # ---- G step: blend + (optional) collapse + adv + anchor ----
            z = seg_forward(m, img)
            with torch.no_grad():
                z0 = seg_forward(teacher, img)
            loss = _masked_ce(z, merged, forget) + lam_r * _masked_kl(z, z0, retain)
            if use_collapse:
                feat = hook.feat
                _, _, h, w = feat.shape
                yl = F.interpolate(orig.float().unsqueeze(1), size=(h, w), mode="nearest").long().squeeze(1)
                ml = F.interpolate(merged.float().unsqueeze(1), size=(h, w), mode="nearest").long().squeeze(1)
                fm = yl == PERSON
                feat_p = feat.permute(0, 2, 3, 1)
                collapse = (((feat_p[fm] - proto[ml[fm].clamp(0, NUM_CLASSES - 1)]) ** 2).sum(1).mean()
                            if fm.any() else feat.sum() * 0.0)
                loss = loss + lam_c * collapse
            dpred = D(field(z))
            fmask = Mlow > 0.5
            if fmask.any():
                loss = loss + lam_adv * bce(dpred[fmask], torch.zeros_like(dpred[fmask]))
            optG.zero_grad(); loss.backward(); optG.step()

            # ---- retain replay: hard CE on full-distribution images ----
            if replay_it is not None:
                try:
                    rimg, rlab = next(replay_it)
                except StopIteration:
                    replay_it = iter(replay_loader); rimg, rlab = next(replay_it)
                rimg, rlab = rimg.to(device), rlab.to(device)
                optG.zero_grad(); ce(seg_forward(m, rimg), rlab).backward(); optG.step()
    if hook is not None:
        hook.close()
    return m


# --------------------------------------------------------------------------- #
# ADVR — the headline method. Discoverable alias for adv_full(lam_c=0): the
# feature-collapse term off, output-adversarial + retain-anchor + retain-replay
# only. Scripts historically called this via `--name ADV_full --lam_c 0.0`; both
# spellings are kept below so existing checkpoints/scripts keep working.
# --------------------------------------------------------------------------- #
def adv_replay(model, loader, device="cuda", **kw):
    """ADVR = adv_full with lam_c=0 (feature-collapse term disabled)."""
    kw.setdefault("lam_c", 0.0)
    return adv_full(model, loader, device=device, **kw)


CANDIDATES = {
    "A2_ft_anchored": ft_anchored,
    "D1_ga_orth": ga_retain_orth,
    "B1_proto_collapse": proto_collapse,
    "ADV_silhouette": adv_silhouette,
    "ADV_full": adv_full,
    "ADVF": adv_full,
    "ADVR": adv_replay,
}


def run_candidate(name, model, loader, device="cuda", **kw):
    return CANDIDATES[name](model, loader, device=device, **kw)
