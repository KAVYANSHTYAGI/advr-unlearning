"""Segmentation model factory.

Two backbones supported:
  "deeplabv3r50"  -- DeepLabV3-ResNet50 (torchvision). Has an aux classifier,
                     returns an OrderedDict. Penultimate features at classifier[4].
  "segformer_b2"  -- SegFormer-B2 (HuggingFace transformers). Returns a dataclass
                     with .logits [N,C,H/4,W/4] (needs bilinear upsample to H,W).
                     No aux loss. Much higher mIoU (~45 on ADE20K) than DeepLab-R50.
                     Used for backbone-transfer generality check.

Pass arch="segformer_b2" to use SegFormer everywhere. The rest of seg/ calls
seg_forward() which handles both transparently.
"""
import torch
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50

from seg.data import NUM_CLASSES


def get_seg_model(num_classes=NUM_CLASSES, pretrained_backbone=True,
                  device="cuda", arch="deeplabv3r50"):
    if arch == "deeplabv3r50":
        weights_backbone = "IMAGENET1K_V1" if pretrained_backbone else None
        model = deeplabv3_resnet50(weights=None, weights_backbone=weights_backbone,
                                   num_classes=num_classes, aux_loss=True)
    elif arch == "segformer_b2":
        from transformers import SegformerForSemanticSegmentation
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b2",
            num_labels=num_classes,
            id2label={i: str(i) for i in range(num_classes)},
            label2id={str(i): i for i in range(num_classes)},
            ignore_mismatched_sizes=True,
        )
    else:
        raise ValueError(f"unknown arch {arch!r}")
    return model.to(device)


def seg_forward(model, x):
    """Return main logits [N,C,H,W]. Handles both DeepLabV3 and SegFormer."""
    if hasattr(model, "config") and "segformer" in model.config.model_type.lower():
        out = model(pixel_values=x).logits          # [N,C,H/4,W/4]
        return F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return model(x)["out"]
