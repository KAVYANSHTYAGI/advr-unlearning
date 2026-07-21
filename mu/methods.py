"""Approximate unlearning methods, ported/wrapped from SalUn and SSD.

Every method takes the original (fully-trained) model plus the data `Splits`
and returns an unlearned copy of the model. The original is never mutated.

Implemented:
  ft        - finetune on the retain set
  ga        - gradient ascent on the forget set (a.k.a. NegGrad)
  neggrad+  - descent on retain + ascent on forget, jointly
  rl        - random-label relabel of forget, then finetune (SalUn base)
  salun     - rl, but only saliency-selected weights are updated
  ssd       - Selective Synaptic Dampening (Fisher-importance dampening)
  scrub     - student-teacher distillation (align retain, diverge forget)
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _clone(model):
    return copy.deepcopy(model)


# --------------------------------------------------------------------------- #
# Finetune-family
# --------------------------------------------------------------------------- #
def ft(model, s, device="cuda", epochs=10, lr=0.01, **_):
    m = _clone(model).to(device).train()
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()
    for _e in range(epochs):
        for x, y in s.retain_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(m(x), y).backward(); opt.step()
    return m


def ga(model, s, device="cuda", epochs=5, lr=1e-4, **_):
    """Gradient ascent on the forget set (maximize loss)."""
    m = _clone(model).to(device).train()
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()
    for _e in range(epochs):
        for x, y in s.forget_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); (-crit(m(x), y)).backward(); opt.step()
    return m


def neggrad_plus(model, s, device="cuda", epochs=5, lr=0.01, alpha=0.999, **_):
    """Joint: descend on retain, ascend on forget (Kurmanji et al. baseline)."""
    m = _clone(model).to(device).train()
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()
    f_iter = iter(s.forget_loader)
    for _e in range(epochs):
        for x, y in s.retain_loader:
            try:
                xf, yf = next(f_iter)
            except StopIteration:
                f_iter = iter(s.forget_loader); xf, yf = next(f_iter)
            x, y, xf, yf = x.to(device), y.to(device), xf.to(device), yf.to(device)
            opt.zero_grad()
            loss = alpha * crit(m(x), y) - (1 - alpha) * crit(m(xf), yf)
            loss.backward(); opt.step()
    return m


# --------------------------------------------------------------------------- #
# Random-label family (RL / SalUn)
# --------------------------------------------------------------------------- #
def _rl_step(m, opt, crit, x, y, num_classes, device):
    # random labels different from the true label
    rand = torch.randint(0, num_classes, y.shape, device=device)
    collide = rand == y
    rand[collide] = (rand[collide] + 1) % num_classes
    opt.zero_grad(); crit(m(x), rand).backward()


def _saliency_mask(model, s, device, threshold=0.5):
    """Gradient saliency mask: per-parameter |grad| on the forget set, kept above
    the `threshold` quantile (SalUn). Returns dict name -> {0,1} mask."""
    m = _clone(model).to(device).train()
    crit = nn.CrossEntropyLoss()
    grads = {n: torch.zeros_like(p) for n, p in m.named_parameters()}
    for x, y in s.forget_loader:
        x, y = x.to(device), y.to(device)
        m.zero_grad(); crit(m(x), y).backward()
        for n, p in m.named_parameters():
            if p.grad is not None:
                grads[n] += p.grad.detach().abs()
    flat = torch.cat([g.flatten() for g in grads.values()])
    cutoff = torch.quantile(flat, threshold)
    return {n: (g >= cutoff).float() for n, g in grads.items()}


def rl(model, s, device="cuda", epochs=10, lr=0.013, mask=None, **_):
    m = _clone(model).to(device).train()
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()
    r_iter = iter(s.retain_loader)
    for _e in range(epochs):
        for x, y in s.forget_loader:
            x, y = x.to(device), y.to(device)
            _rl_step(m, opt, crit, x, y, s.num_classes, device)
            if mask is not None:
                for n, p in m.named_parameters():
                    if p.grad is not None:
                        p.grad *= mask[n]
            opt.step()
            # one retain batch to preserve utility
            try:
                xr, yr = next(r_iter)
            except StopIteration:
                r_iter = iter(s.retain_loader); xr, yr = next(r_iter)
            xr, yr = xr.to(device), yr.to(device)
            opt.zero_grad(); crit(m(xr), yr).backward()
            if mask is not None:
                for n, p in m.named_parameters():
                    if p.grad is not None:
                        p.grad *= mask[n]
            opt.step()
    return m


def salun(model, s, device="cuda", epochs=10, lr=0.013, threshold=0.5, **_):
    mask = _saliency_mask(model, s, device, threshold=threshold)
    return rl(model, s, device=device, epochs=epochs, lr=lr, mask=mask)


# --------------------------------------------------------------------------- #
# SSD (Selective Synaptic Dampening) - ported from vendor/SSD/src/ssd.py
# --------------------------------------------------------------------------- #
def _ssd_importance(model, loader, device):
    crit = nn.CrossEntropyLoss()
    imp = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model.zero_grad(); crit(model(x), y).backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                imp[n] += p.grad.detach().pow(2)
    for n in imp:
        imp[n] /= float(len(loader))
    return imp


def ssd(model, s, device="cuda", dampening_constant=1.0, selection_weighting=10.0,
        exponent=1.0, lower_bound=1.0, **_):
    m = _clone(model).to(device)
    # importance must use CLEAN (non-augmented) data -- augmentation makes the
    # estimate RNG-dependent and can push SSD into a destructive regime.
    full_imp = _ssd_importance(m, s.full_eval, device)
    forget_imp = _ssd_importance(m, s.forget_eval, device)
    with torch.no_grad():
        for (n, p) in m.named_parameters():
            oimp, fimp = full_imp[n], forget_imp[n]
            loc = torch.where(fimp > oimp.mul(selection_weighting))
            weight = (oimp.mul(dampening_constant).div(fimp + 1e-12)).pow(exponent)
            upd = weight[loc]
            upd[upd > lower_bound] = lower_bound
            p[loc] = p[loc].mul(upd)
    return m


# --------------------------------------------------------------------------- #
# SCRUB (Kurmanji et al., NeurIPS 2023)
# --------------------------------------------------------------------------- #
def scrub(model, s, device="cuda", epochs=5, lr=5e-4, kd_T=4.0,
          alpha=0.5, gamma=1.0, msteps=2, **_):
    student = _clone(model).to(device).train()
    teacher = _clone(model).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    ce = nn.CrossEntropyLoss()

    def kl(s_logits, t_logits):
        return F.kl_div(F.log_softmax(s_logits / kd_T, 1),
                        F.softmax(t_logits / kd_T, 1),
                        reduction="batchmean") * (kd_T ** 2)

    for e in range(epochs):
        # max-step: push student AWAY from teacher on forget
        if e < msteps:
            for x, _y in s.forget_loader:
                x = x.to(device)
                opt.zero_grad()
                loss = -kl(student(x), teacher(x))
                loss.backward(); opt.step()
        # min-step: keep student CLOSE to teacher + correct on retain
        for x, y in s.retain_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = student(x)
            loss = gamma * ce(out, y) + alpha * kl(out, teacher(x))
            loss.backward(); opt.step()
    return student


METHODS = {
    "ft": ft,
    "ga": ga,
    "neggrad+": neggrad_plus,
    "rl": rl,
    "salun": salun,
    "ssd": ssd,
    "scrub": scrub,
}


def run_method(name, model, s, device="cuda", **kw):
    if name not in METHODS:
        raise ValueError(f"unknown method {name!r}; have {list(METHODS)}")
    return METHODS[name](model, s, device=device, **kw)
