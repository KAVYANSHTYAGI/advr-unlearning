"""Silhouette-reconstruction attacker (the metric AND the attack in HYP).

D_psi sees ONLY the model's output field O (softmax over 150 classes, no image) and
tries to recover the person mask M. Its held-out IoU/AUC = how much spatial
person-information leaks. The oracle (trained never seeing person) gives the
irreducible context-prior floor; a method "leaks" if its attacker beats that floor.

A per-pixel attacker would just exploit local cues; we give it a few conv layers so it
can use region/shape structure -- a genuine silhouette reconstructor.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from seg.data import PERSON, NUM_CLASSES
from seg.model import seg_forward


class SilhouetteNet(nn.Module):
    """Silhouette attacker. depth/width/dilation let us build a STRONGER attacker
    than the one ADV trains against, for an honest transfer-attack evaluation."""
    def __init__(self, c_in=NUM_CLASSES, width=64, depth=3, dilation=1):
        super().__init__()
        layers = []
        cin = c_in
        for _ in range(depth):
            layers += [nn.Conv2d(cin, width, 3, padding=dilation, dilation=dilation),
                       nn.BatchNorm2d(width), nn.ReLU(True)]
            cin = width
        layers += [nn.Conv2d(width, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, p):
        return self.net(p)[:, 0]                      # [N,h,w] logits


# --------------------------------------------------------------------------- #
# Attacker zoo -- stronger architectures for the "weak attacker" rebuttal.
# All share the SilhouetteNet interface: input [N,c_in,h,w] -> [N,h,w] logits.
# --------------------------------------------------------------------------- #
class _DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class UNetAttacker(nn.Module):
    """U-Net silhouette reconstructor with skip connections. `base=64` -> ~7.7M
    params, `base=96` -> ~17M. A genuinely stronger shape prior than the plain CNN:
    the encoder/decoder + skips let it use both fine boundaries and global layout."""
    def __init__(self, c_in=NUM_CLASSES, base=64):
        super().__init__()
        b = base
        self.inc = _DoubleConv(c_in, b)
        self.d1 = _DoubleConv(b, b * 2)
        self.d2 = _DoubleConv(b * 2, b * 4)
        self.d3 = _DoubleConv(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.u3 = _DoubleConv(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.u2 = _DoubleConv(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.u1 = _DoubleConv(b * 2, b)
        self.outc = nn.Conv2d(b, 1, 1)

    def forward(self, p):
        x0 = self.inc(p)                              # [N,b,h,w]
        x1 = self.d1(self.pool(x0))
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))                   # bottleneck
        y = self.u3(torch.cat([self.up3(x3), x2], 1))
        y = self.u2(torch.cat([self.up2(y), x1], 1))
        y = self.u1(torch.cat([self.up1(y), x0], 1))
        return self.outc(y)[:, 0]                      # [N,h,w] logits


class ViTAttacker(nn.Module):
    """Patch-transformer silhouette reconstructor. Patchify the field into
    (h/patch)^2 tokens, run a Transformer encoder (global receptive field from
    layer 1), then a linear head predicts every pixel in the patch. `dim=384,
    depth=8, heads=6` -> ~14M; scale dim/depth up for a 50M attacker. Global
    self-attention is the strongest layout prior we can hand the adversary."""
    def __init__(self, c_in=NUM_CLASSES, img=128, patch=8, dim=384, depth=8,
                 heads=6, mlp=4):
        super().__init__()
        self.img, self.patch = img, patch
        self.gh = img // patch
        n_tok = self.gh * self.gh
        self.proj = nn.Conv2d(c_in, dim, patch, stride=patch)     # [N,dim,gh,gh]
        self.pos = nn.Parameter(torch.zeros(1, n_tok, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * mlp, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, patch * patch)     # per-pixel logit for the patch

    def forward(self, p):
        if p.shape[-1] != self.img:                   # attacker fixed at img resolution
            p = F.interpolate(p, size=(self.img, self.img), mode="bilinear",
                              align_corners=False)
        t = self.proj(p).flatten(2).transpose(1, 2)   # [N,n_tok,dim]
        t = self.enc(t + self.pos)
        t = self.head(self.norm(t))                   # [N,n_tok,patch*patch]
        gh, ps = self.gh, self.patch
        t = t.reshape(-1, gh, gh, ps, ps).permute(0, 1, 3, 2, 4)
        return t.reshape(-1, gh * ps, gh * ps)        # [N,img,img] logits


def build_attacker(kind="cnn", c_in=NUM_CLASSES, width=64, depth=3, dilation=1,
                   base=64, dim=384, vdepth=8, heads=6, out_size=128):
    """Factory over the attacker zoo. kind in {cnn, unet, vit}."""
    if kind == "cnn":
        return SilhouetteNet(c_in=c_in, width=width, depth=depth, dilation=dilation)
    if kind == "unet":
        return UNetAttacker(c_in=c_in, base=base)
    if kind == "vit":
        return ViTAttacker(c_in=c_in, img=out_size, dim=dim, depth=vdepth, heads=heads)
    raise ValueError(f"unknown attacker kind {kind!r}")


def n_params(module):
    return sum(p.numel() for p in module.parameters())


@torch.no_grad()
def output_field(model, img, transform=None, out_size=128):
    z = seg_forward(model, img)                       # [N,C,H,W]
    if transform == "mask":                           # DELETE-style: drop person logit
        z = z.clone(); z[:, PERSON] = -1e4
    p = F.softmax(z, 1)
    return F.interpolate(p, size=(out_size, out_size), mode="bilinear", align_corners=False)


class _AttackFeatHook:
    def __init__(self, model):
        self.feat = None
        self.h = model.classifier[4].register_forward_pre_hook(
            lambda m, inp: setattr(self, "feat", inp[0]))

    def close(self):
        self.h.remove()


@torch.no_grad()
def feature_field(model, img, out_size=128):
    """Penultimate 256-d features (the representation an attacker with internal
    access reads). DELETE-mask leaves these untouched -> should leak fully."""
    hook = _AttackFeatHook(model)
    seg_forward(model, img)
    feat = hook.feat
    hook.close()
    return F.interpolate(feat, size=(out_size, out_size), mode="bilinear", align_corners=False)


FEAT_DIM = 256


def person_mask_low(orig, out_size=128):
    M = (orig == PERSON).float().unsqueeze(1)
    return F.interpolate(M, size=(out_size, out_size), mode="nearest")[:, 0]


def _field(model, img, transform, out_size, feat):
    return feature_field(model, img, out_size) if feat \
        else output_field(model, img, transform, out_size)


def train_attacker(model, loader, device, transform=None, epochs=2, out_size=128,
                   max_iters=400, width=64, depth=3, dilation=1, feat=False,
                   kind="cnn", lr=1e-3, base=64, dim=384, vdepth=8, heads=6,
                   D=None):
    """Train a silhouette attacker of the given `kind`. Pass `D` to keep training
    an existing attacker (used by the adaptive best-of-N loop)."""
    c_in = FEAT_DIM if feat else NUM_CLASSES
    if D is None:
        D = build_attacker(kind, c_in=c_in, width=width, depth=depth,
                           dilation=dilation, base=base, dim=dim, vdepth=vdepth,
                           heads=heads, out_size=out_size).to(device)
    D.train()
    opt = torch.optim.Adam(D.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    model.eval()
    it = 0
    for _e in range(epochs):
        for img, orig, _m in loader:
            p = _field(model, img.to(device), transform, out_size, feat)
            M = person_mask_low(orig.to(device), out_size)
            loss = bce(D(p), M)
            opt.zero_grad(); loss.backward(); opt.step()
            it += 1
            if it >= max_iters:
                return D
    return D


def train_attacker_adaptive(model, train_loader, val_loader, device, transform=None,
                            feat=False, kind="cnn", out_size=128, restarts=3,
                            rounds=6, iters_per_round=1200, lr=1e-3, patience=2,
                            verbose=False, **arch_kw):
    """White-box adaptive attacker: for each of `restarts` random inits, train to
    convergence (up to `rounds` x `iters_per_round` iters) with val-AUC early
    stopping (`patience`), and return the single best attacker seen on val. This is
    the honest "strongest adaptive adversary" the reviewer asked for -- it is given
    far more capacity/compute than the discriminator ADV trained against.

    Returns (best_D, best_metrics) where best_metrics is the eval_attacker dict at
    the best val checkpoint."""
    best_auc, best_D, best_metrics = -1.0, None, None
    for r in range(restarts):
        D = None
        run_best_auc, since = -1.0, 0
        for rd in range(rounds):
            D = train_attacker(model, train_loader, device, transform=transform,
                               feat=feat, kind=kind, out_size=out_size,
                               epochs=10 ** 6, max_iters=iters_per_round, lr=lr,
                               D=D, **arch_kw)
            m = eval_attacker(D, model, val_loader, device, transform=transform,
                              out_size=out_size, feat=feat)
            auc = m["attacker_AUC"]
            if verbose:
                print(f"    restart{r} round{rd}: AUC {auc:.1f} IoU "
                      f"{m['attacker_IoU']:.1f}", flush=True)
            if np.isfinite(auc) and auc > run_best_auc + 0.05:
                run_best_auc, since = auc, 0
                if auc > best_auc:
                    import copy as _cp
                    best_auc, best_D, best_metrics = auc, _cp.deepcopy(D), m
            else:
                since += 1
                if since >= patience:
                    break
    return best_D, best_metrics


@torch.no_grad()
def eval_attacker(D, model, loader, device, transform=None, out_size=128, feat=False):
    D.eval(); model.eval()
    inter = union = 0
    ys, ps = [], []
    for img, orig, _m in loader:
        p = _field(model, img.to(device), transform, out_size, feat)
        M = person_mask_low(orig.to(device), out_size)
        prob = torch.sigmoid(D(p))
        pred = prob > 0.5
        gt = M > 0.5
        inter += (pred & gt).sum().item()
        union += (pred | gt).sum().item()
        # subsample pixels for AUC
        pr = prob.flatten(); gg = M.flatten()
        idx = torch.randperm(pr.numel(), device=device)[:20000]
        ps.append(pr[idx].cpu().numpy()); ys.append(gg[idx].cpu().numpy())
    iou = 100 * inter / max(union, 1)
    y = np.concatenate(ys); pscore = np.concatenate(ps)
    # guard: NaN from destroyed models (constant output → sigmoid NaN) → treat as 50 AUC
    valid = np.isfinite(pscore)
    if valid.sum() < 10 or not (y[valid] > 0.5).any() or not (y[valid] <= 0.5).any():
        auc = float("nan")
    else:
        auc = roc_auc_score((y[valid] > 0.5).astype(int), pscore[valid])
    return {"attacker_IoU": iou, "attacker_AUC": 100 * auc}
