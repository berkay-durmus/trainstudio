"""The curated model catalogue.

Sorted by `released` in descending order, the most recent architectures in the
literature land at the top of the list — that is the default ordering of the
Model Selection page.

The `rec` field carries model-specific hyperparameter recommendations;
core/recommend.py blends those with dataset statistics and the local hardware.

The timm/smp/ultralytics names are real library identifiers — the `arch` field is
passed straight through to the corresponding backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.schemas import Backend, Task

# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelSpec:
    id: str                              # registry key (unique)
    display_name: str
    task: Task
    backend: Backend
    arch: str                            # the real name passed to the backend
    family: str                          # for grouping/filtering
    released: str                        # "YYYY-MM" — sort key
    params_m: float                      # parameters in millions
    default_img_size: int
    min_vram_gb: float                   # rough figure, for batch=8
    size_class: str                      # nano | small | base | large
    summary: str                         # one-sentence description
    strengths: tuple[str, ...] = ()
    gflops: float | None = None
    weights: str | None = None           # default pretrained tag/weights
    needs_encoder: bool = False          # smp architectures require an encoder
    paper_url: str = ""
    rec: dict[str, Any] = field(default_factory=dict)

    @property
    def year(self) -> int:
        return int(self.released.split("-")[0])

    @property
    def is_recent(self) -> bool:
        """Drives the 'NEW' badge in the catalogue — roughly the last 2 years."""
        return self.year >= 2025

    @property
    def params_label(self) -> str:
        if self.params_m >= 1000:
            return f"{self.params_m / 1000:.1f}B"
        if self.params_m >= 10:
            return f"{self.params_m:.0f}M"
        return f"{self.params_m:.1f}M"


# Short aliases — so the tables below read like tables
CLS, SEG, SEG3D = Task.CLASSIFICATION, Task.SEGMENTATION, Task.SEGMENTATION3D
TIMM, TV, SMP, HF, ULTRA, MONAI = (
    Backend.TIMM, Backend.TORCHVISION, Backend.SMP, Backend.HF,
    Backend.ULTRALYTICS, Backend.MONAI,
)

# Frequently reused recommendation blocks
_VIT_LARGE = dict(optimizer="adamw", lr=5e-5, weight_decay=0.05, layer_decay=0.75,
                  label_smoothing=0.1, grad_clip=1.0, warmup_epochs=5, drop_path_rate=0.2)
_VIT_BASE = dict(optimizer="adamw", lr=1e-4, weight_decay=0.05, layer_decay=0.65,
                 label_smoothing=0.1, grad_clip=1.0, warmup_epochs=3, drop_path_rate=0.1)
_VIT_SMALL = dict(optimizer="adamw", lr=2e-4, weight_decay=0.05, label_smoothing=0.1,
                  grad_clip=1.0, warmup_epochs=3, drop_path_rate=0.1)
_CNN_MODERN = dict(optimizer="adamw", lr=5e-4, weight_decay=0.05, label_smoothing=0.1,
                   warmup_epochs=3, drop_path_rate=0.1)
_CNN_CLASSIC = dict(optimizer="sgd", lr=1e-2, weight_decay=1e-4, momentum=0.9,
                    warmup_epochs=2, scheduler="cosine")
_SEG_DEFAULT = dict(optimizer="adamw", lr=3e-4, weight_decay=1e-4,
                    loss="dice_ce", warmup_epochs=2, scheduler="cosine")
_YOLO = dict(optimizer="adamw", lr=1e-3, weight_decay=5e-4, warmup_epochs=3,
             scheduler="cosine")


# ═════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION — timm
# ═════════════════════════════════════════════════════════════════════════════

_CLS_TIMM = [
    ModelSpec(
        id="dinov3_vit_l16", display_name="DINOv3 ViT-L/16", task=CLS, backend=TIMM,
        arch="vit_large_patch16_dinov3.lvd1689m", family="DINOv3", released="2025-08",
        params_m=304, default_img_size=224, min_vram_gb=18, size_class="large",
        summary="Meta's strongest general-purpose visual representation, self-supervised on LVD-1689M.",
        strengths=("Strongest transfer", "High accuracy with few labels", "Strong on medical imaging"),
        paper_url="https://github.com/facebookresearch/dinov3",
        rec=dict(_VIT_LARGE, batch_size=8, freeze_backbone_epochs=2),
    ),
    ModelSpec(
        id="dinov3_vit_b16", display_name="DINOv3 ViT-B/16", task=CLS, backend=TIMM,
        arch="vit_base_patch16_dinov3.lvd1689m", family="DINOv3", released="2025-08",
        params_m=86, default_img_size=224, min_vram_gb=9, size_class="base",
        summary="The best accuracy/cost trade-off in the DINOv3 family — the first choice for most projects.",
        strengths=("Strong transfer", "Reasonable cost", "Good with little data"),
        paper_url="https://github.com/facebookresearch/dinov3",
        rec=dict(_VIT_BASE, batch_size=16, freeze_backbone_epochs=2),
    ),
    ModelSpec(
        id="dinov3_vit_s16", display_name="DINOv3 ViT-S/16", task=CLS, backend=TIMM,
        arch="vit_small_patch16_dinov3.lvd1689m", family="DINOv3", released="2025-08",
        params_m=21, default_img_size=224, min_vram_gb=5, size_class="small",
        summary="The lightweight DINOv3; more resistant to overfitting on small datasets.",
        strengths=("Fast", "Suited to small datasets"),
        paper_url="https://github.com/facebookresearch/dinov3",
        rec=dict(_VIT_SMALL, batch_size=32),
    ),
    ModelSpec(
        id="dinov3_convnext_b", display_name="DINOv3 ConvNeXt-Base", task=CLS, backend=TIMM,
        arch="convnext_base.dinov3_lvd1689m", family="DINOv3", released="2025-08",
        params_m=89, default_img_size=224, min_vram_gb=9, size_class="base",
        summary="A convolutional backbone distilled with DINOv3 — cheaper than a ViT at high resolution.",
        strengths=("Convolutional", "Efficient at high resolution"),
        paper_url="https://github.com/facebookresearch/dinov3",
        rec=dict(_CNN_MODERN, lr=2e-4, batch_size=16),
    ),
    ModelSpec(
        id="dinov3_convnext_t", display_name="DINOv3 ConvNeXt-Tiny", task=CLS, backend=TIMM,
        arch="convnext_tiny.dinov3_lvd1689m", family="DINOv3", released="2025-08",
        params_m=29, default_img_size=224, min_vram_gb=5, size_class="small",
        summary="A light DINOv3-distilled ConvNeXt; ideal for fast experiment rounds.",
        strengths=("Fast", "Low memory"),
        rec=dict(_CNN_MODERN, batch_size=32),
    ),
    ModelSpec(
        id="mobilenetv4_l", display_name="MobileNetV4 Conv-Large", task=CLS, backend=TIMM,
        arch="mobilenetv4_conv_large.e600_r384_in1k", family="MobileNetV4", released="2024-05",
        params_m=32, default_img_size=384, min_vram_gb=5, size_class="small",
        summary="A modern mobile architecture designed for edge devices, fast on virtually any hardware.",
        strengths=("Edge devices", "Very fast inference"),
        rec=dict(_CNN_MODERN, batch_size=32),
    ),
    ModelSpec(
        id="mobilenetv4_hybrid_m", display_name="MobileNetV4 Hybrid-Medium", task=CLS, backend=TIMM,
        arch="mobilenetv4_hybrid_medium.ix_e550_r256_in1k", family="MobileNetV4", released="2024-05",
        params_m=11, default_img_size=256, min_vram_gb=4, size_class="nano",
        summary="A light convolution + attention hybrid; built for real-time scenarios.",
        strengths=("Very light", "Real time"),
        rec=dict(_CNN_MODERN, batch_size=64),
    ),
    ModelSpec(
        id="hiera_base", display_name="Hiera-Base", task=CLS, backend=TIMM,
        arch="hiera_base_224.mae_in1k_ft_in1k", family="Hiera", released="2023-06",
        params_m=51, default_img_size=224, min_vram_gb=7, size_class="base",
        summary="A hierarchical ViT trained with MAE and stripped of its complex parts — simple and fast.",
        strengths=("Hierarchical", "MAE pretrained"),
        rec=dict(_VIT_BASE, batch_size=16),
    ),
    ModelSpec(
        id="fastvit_sa24", display_name="FastViT-SA24", task=CLS, backend=TIMM,
        arch="fastvit_sa24.apple_in1k", family="FastViT", released="2023-03",
        params_m=21, default_img_size=256, min_vram_gb=5, size_class="small",
        summary="Apple's structurally reparameterised hybrid; strong on mobile latency.",
        strengths=("Low latency", "Hybrid"),
        rec=dict(_CNN_MODERN, batch_size=32),
    ),
    ModelSpec(
        id="eva02_l14", display_name="EVA-02 Large/14", task=CLS, backend=TIMM,
        arch="eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", family="EVA-02", released="2023-03",
        params_m=305, default_img_size=448, min_vram_gb=24, size_class="large",
        summary="A MIM-based ViT that held the ImageNet top spot for a long time; for maximum accuracy.",
        strengths=("Very high accuracy", "448px input"),
        rec=dict(_VIT_LARGE, batch_size=4, img_size=448),
    ),
    ModelSpec(
        id="eva02_b14", display_name="EVA-02 Base/14", task=CLS, backend=TIMM,
        arch="eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", family="EVA-02", released="2023-03",
        params_m=87, default_img_size=448, min_vram_gb=12, size_class="base",
        summary="The balanced EVA-02; for work that needs fine detail at high resolution.",
        strengths=("High accuracy", "Fine detail"),
        rec=dict(_VIT_BASE, batch_size=8, img_size=448),
    ),
    ModelSpec(
        id="convnextv2_base", display_name="ConvNeXt V2 Base", task=CLS, backend=TIMM,
        arch="convnextv2_base.fcmae_ft_in22k_in1k", family="ConvNeXt", released="2023-01",
        params_m=89, default_img_size=224, min_vram_gb=9, size_class="base",
        summary="An FCMAE-pretrained modern convolutional network; competitive with transformers and more stable to train.",
        strengths=("Stable training", "Strong baseline"),
        rec=dict(_CNN_MODERN, batch_size=16),
    ),
    ModelSpec(
        id="convnextv2_tiny", display_name="ConvNeXt V2 Tiny", task=CLS, backend=TIMM,
        arch="convnextv2_tiny.fcmae_ft_in22k_in1k", family="ConvNeXt", released="2023-01",
        params_m=29, default_img_size=224, min_vram_gb=5, size_class="small",
        summary="The lightweight ConvNeXt V2; a very good balance on small and mid-sized datasets.",
        strengths=("Balanced", "Fast"),
        rec=dict(_CNN_MODERN, batch_size=32),
    ),
    ModelSpec(
        id="maxvit_tiny", display_name="MaxViT-Tiny", task=CLS, backend=TIMM,
        arch="maxvit_tiny_tf_224.in1k", family="MaxViT", released="2022-04",
        params_m=31, default_img_size=224, min_vram_gb=6, size_class="small",
        summary="A multi-scale hybrid combining local and global attention.",
        strengths=("Multi-scale attention",),
        rec=dict(_VIT_SMALL, batch_size=24),
    ),
    ModelSpec(
        id="swinv2_base", display_name="Swin Transformer V2 Base", task=CLS, backend=TIMM,
        arch="swinv2_base_window8_256.ms_in1k", family="Swin", released="2021-11",
        params_m=88, default_img_size=256, min_vram_gb=10, size_class="base",
        summary="A hierarchical transformer with shifted-window attention; widely used for dense prediction.",
        strengths=("Hierarchical", "Widely adopted"),
        rec=dict(_VIT_BASE, batch_size=16, img_size=256),
    ),
    ModelSpec(
        id="efficientnetv2_s", display_name="EfficientNetV2-S", task=CLS, backend=TIMM,
        arch="tf_efficientnetv2_s.in21k_ft_in1k", family="EfficientNet", released="2021-04",
        params_m=22, default_img_size=384, min_vram_gb=6, size_class="small",
        summary="A fast-training convolutional network with high accuracy per parameter.",
        strengths=("Fast training", "Parameter efficient"),
        rec=dict(_CNN_MODERN, lr=1e-3, batch_size=32, img_size=384),
    ),
    ModelSpec(
        id="vit_b16", display_name="ViT-B/16", task=CLS, backend=TIMM,
        arch="vit_base_patch16_224.augreg2_in21k_ft_in1k", family="ViT", released="2020-10",
        params_m=86, default_img_size=224, min_vram_gb=9, size_class="base",
        summary="The original Vision Transformer; the standard reference in comparison studies.",
        strengths=("Reference model", "Well documented"),
        rec=dict(_VIT_BASE, batch_size=16),
    ),
    ModelSpec(
        id="regnety_032", display_name="RegNetY-3.2GF", task=CLS, backend=TIMM,
        arch="regnety_032.ra_in1k", family="RegNet", released="2020-03",
        params_m=19, default_img_size=224, min_vram_gb=5, size_class="small",
        summary="A convolutional network found by design-space search that scales predictably.",
        strengths=("Predictable", "Efficient"),
        rec=dict(_CNN_CLASSIC, batch_size=32),
    ),
    ModelSpec(
        id="efficientnet_b0", display_name="EfficientNet-B0", task=CLS, backend=TIMM,
        arch="efficientnet_b0.ra_in1k", family="EfficientNet", released="2019-05",
        params_m=5.3, default_img_size=224, min_vram_gb=4, size_class="nano",
        summary="Classic compound scaling; ideal for establishing a quick baseline.",
        strengths=("Very fast", "Small"),
        rec=dict(_CNN_MODERN, lr=1e-3, batch_size=64),
    ),
    ModelSpec(
        id="densenet121", display_name="DenseNet-121", task=CLS, backend=TIMM,
        arch="densenet121.ra_in1k", family="DenseNet", released="2016-08",
        params_m=8.0, default_img_size=224, min_vram_gb=5, size_class="nano",
        summary="A classic densely connected network; the most common baseline in the medical imaging literature.",
        strengths=("Standard in medical literature", "Few parameters"),
        rec=dict(_CNN_CLASSIC, lr=1e-2, batch_size=32),
    ),
    ModelSpec(
        id="resnet50", display_name="ResNet-50", task=CLS, backend=TIMM,
        arch="resnet50.a1_in1k", family="ResNet", released="2015-12",
        params_m=25.6, default_img_size=224, min_vram_gb=5, size_class="small",
        summary="Computer vision's most widely used reference model; the comparison baseline.",
        strengths=("Universal reference", "Fast"),
        rec=dict(_CNN_CLASSIC, batch_size=32),
    ),
    ModelSpec(
        id="resnet101", display_name="ResNet-101", task=CLS, backend=TIMM,
        arch="resnet101.a1_in1k", family="ResNet", released="2015-12",
        params_m=44.5, default_img_size=224, min_vram_gb=7, size_class="base",
        summary="A deeper ResNet; gains over ResNet-50 on large datasets.",
        strengths=("Deeper", "Reference"),
        rec=dict(_CNN_CLASSIC, batch_size=24),
    ),
    ModelSpec(
        id="resnet152", display_name="ResNet-152", task=CLS, backend=TIMM,
        arch="resnet152.a1_in1k", family="ResNet", released="2015-12",
        params_m=60.2, default_img_size=224, min_vram_gb=9, size_class="base",
        summary="The deepest standard member of the ResNet family.",
        strengths=("Deepest ResNet",),
        rec=dict(_CNN_CLASSIC, batch_size=16),
    ),
]

# ═════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION — Ultralytics
# ═════════════════════════════════════════════════════════════════════════════

_CLS_ULTRA = [
    ModelSpec(
        id="yolo26s_cls", display_name="YOLO26-s (cls)", task=CLS, backend=ULTRA,
        arch="yolo26s-cls.pt", family="YOLO", released="2026-01",
        params_m=6.0, default_img_size=224, min_vram_gb=5, size_class="small",
        summary="The classification head of the newest Ultralytics release; fast end to end.",
        strengths=("Newest YOLO", "Fast inference", "Easy export"),
        paper_url="https://docs.ultralytics.com/models/yolo26",
        rec=dict(_YOLO, batch_size=32),
    ),
    ModelSpec(
        id="yolo26n_cls", display_name="YOLO26-n (cls)", task=CLS, backend=ULTRA,
        arch="yolo26n-cls.pt", family="YOLO", released="2026-01",
        params_m=2.5, default_img_size=224, min_vram_gb=4, size_class="nano",
        summary="The smallest YOLO26 classifier; for edge devices and quick prototypes.",
        strengths=("Very light", "Edge devices"),
        rec=dict(_YOLO, batch_size=64),
    ),
    ModelSpec(
        id="yolo11s_cls", display_name="YOLO11-s (cls)", task=CLS, backend=ULTRA,
        arch="yolo11s-cls.pt", family="YOLO", released="2024-09",
        params_m=5.5, default_img_size=224, min_vram_gb=5, size_class="small",
        summary="The widely validated previous-generation YOLO classifier.",
        strengths=("Mature", "Large community"),
        rec=dict(_YOLO, batch_size=32),
    ),
    ModelSpec(
        id="yolov8s_cls", display_name="YOLOv8-s (cls)", task=CLS, backend=ULTRA,
        arch="yolov8s-cls.pt", family="YOLO", released="2023-01",
        params_m=6.4, default_img_size=224, min_vram_gb=5, size_class="small",
        summary="The YOLOv8 classification head, kept for backwards comparison.",
        strengths=("Comparison baseline",),
        rec=dict(_YOLO, batch_size=32),
    ),
]

# ═════════════════════════════════════════════════════════════════════════════
# SEGMENTATION 2D — Ultralytics
# ═════════════════════════════════════════════════════════════════════════════

_SEG_ULTRA = [
    ModelSpec(
        id="yolo26s_seg", display_name="YOLO26-s (instance seg)", task=SEG, backend=ULTRA,
        arch="yolo26s-seg.pt", family="YOLO", released="2026-01",
        params_m=10.0, default_img_size=640, min_vram_gb=7, size_class="small",
        summary="The instance-segmentation head of the newest YOLO; for work that needs objects separated.",
        strengths=("Newest YOLO", "Instance separation", "Real time"),
        paper_url="https://docs.ultralytics.com/models/yolo26",
        rec=dict(_YOLO, batch_size=16, img_size=640),
    ),
    ModelSpec(
        id="yolo26n_seg", display_name="YOLO26-n (instance seg)", task=SEG, backend=ULTRA,
        arch="yolo26n-seg.pt", family="YOLO", released="2026-01",
        params_m=3.4, default_img_size=640, min_vram_gb=5, size_class="nano",
        summary="Lightweight YOLO26 instance segmentation; for quick experiments and edge devices.",
        strengths=("Very light", "Fast"),
        rec=dict(_YOLO, batch_size=32, img_size=640),
    ),
    ModelSpec(
        id="yolo26s_sem", display_name="YOLO26-s (semantic seg)", task=SEG, backend=ULTRA,
        arch="yolo26s-sem.pt", family="YOLO", released="2026-01",
        params_m=10.0, default_img_size=640, min_vram_gb=7, size_class="small",
        summary="YOLO26's native semantic segmentation head — masks are used directly, never converted to polygons.",
        strengths=("Native semantic", "No polygon conversion"),
        rec=dict(_YOLO, batch_size=16, img_size=640),
    ),
    ModelSpec(
        id="yolo11s_seg", display_name="YOLO11-s (instance seg)", task=SEG, backend=ULTRA,
        arch="yolo11s-seg.pt", family="YOLO", released="2024-09",
        params_m=10.1, default_img_size=640, min_vram_gb=7, size_class="small",
        summary="Mature, widely validated previous-generation YOLO instance segmentation.",
        strengths=("Mature", "Large community"),
        rec=dict(_YOLO, batch_size=16, img_size=640),
    ),
    ModelSpec(
        id="yolov8s_seg", display_name="YOLOv8-s (instance seg)", task=SEG, backend=ULTRA,
        arch="yolov8s-seg.pt", family="YOLO", released="2023-01",
        params_m=11.8, default_img_size=640, min_vram_gb=7, size_class="small",
        summary="YOLOv8 instance segmentation as a comparison baseline.",
        strengths=("Comparison baseline",),
        rec=dict(_YOLO, batch_size=16, img_size=640),
    ),
]

# ═════════════════════════════════════════════════════════════════════════════
# SEGMENTATION 2D — HuggingFace transformers
# ═════════════════════════════════════════════════════════════════════════════

_SEG_HF = [
    ModelSpec(
        id="hf_mask2former_base", display_name="Mask2Former (Swin-B)", task=SEG, backend=HF,
        arch="facebook/mask2former-swin-base-ade-semantic", family="Mask2Former", released="2021-12",
        params_m=107, default_img_size=512, min_vram_gb=14, size_class="large",
        summary="Universal segmentation based on masked attention; among the strongest on complex scenes.",
        strengths=("Universal segmentation", "High accuracy"),
        paper_url="https://arxiv.org/abs/2112.01527",
        rec=dict(_SEG_DEFAULT, lr=5e-5, batch_size=4, img_size=512, loss="native"),
    ),
    ModelSpec(
        id="hf_segformer_b2", display_name="SegFormer-B2 (HF)", task=SEG, backend=HF,
        arch="nvidia/segformer-b2-finetuned-ade-512-512", family="SegFormer", released="2021-05",
        params_m=27.4, default_img_size=512, min_vram_gb=8, size_class="base",
        summary="A transformer with a lightweight MLP decoder; a very good accuracy/speed balance.",
        strengths=("Efficient", "No positional encoding needed"),
        paper_url="https://arxiv.org/abs/2105.15203",
        rec=dict(_SEG_DEFAULT, lr=6e-5, batch_size=8, img_size=512, loss="native"),
    ),
    ModelSpec(
        id="hf_segformer_b0", display_name="SegFormer-B0 (HF)", task=SEG, backend=HF,
        arch="nvidia/segformer-b0-finetuned-ade-512-512", family="SegFormer", released="2021-05",
        params_m=3.8, default_img_size=512, min_vram_gb=5, size_class="nano",
        summary="The lightest SegFormer; for a fast baseline and edge devices.",
        strengths=("Very light", "Fast"),
        rec=dict(_SEG_DEFAULT, lr=6e-5, batch_size=16, img_size=512, loss="native"),
    ),
]

# ═════════════════════════════════════════════════════════════════════════════
# SEGMENTATION 2D — segmentation_models_pytorch (encoder is selectable)
# ═════════════════════════════════════════════════════════════════════════════

_SMP_ROWS = [
    # (id, name, arch, released, params_m, vram, size, summary, strengths, extra recommendation)
    ("smp_dpt", "DPT", "DPT", "2021-03", 123, 14, "large",
     "A dense prediction transformer; carries global context very well.",
     ("Global context", "High accuracy"), dict(lr=1e-4, batch_size=4)),
    ("smp_segformer", "SegFormer (smp)", "Segformer", "2021-05", 27.4, 8, "base",
     "The SegFormer architecture as implemented in smp; can be paired with any timm encoder.",
     ("Efficient", "Flexible encoder"), dict(lr=2e-4, batch_size=8)),
    ("smp_upernet", "UPerNet", "UPerNet", "2018-07", 60, 10, "base",
     "Pyramid pooling combined with an FPN; strong on multi-scale objects.",
     ("Multi-scale", "Robust"), dict(batch_size=8)),
    ("smp_deeplabv3plus", "DeepLabV3+", "DeepLabV3Plus", "2018-02", 45, 9, "base",
     "Atrous convolution plus a decoder; one of segmentation's most dependable references.",
     ("Sharp boundaries", "Dependable reference"), dict(batch_size=8)),
    ("smp_pan", "PAN", "PAN", "2018-05", 35, 7, "small",
     "Pyramid attention network; light and fast.",
     ("Light", "Fast"), dict(batch_size=16)),
    ("smp_manet", "MA-Net", "MAnet", "2020-08", 40, 8, "base",
     "A U-Net derivative strengthened with multi-scale attention blocks.",
     ("Attention based",), dict(batch_size=8)),
    ("smp_deeplabv3", "DeepLabV3", "DeepLabV3", "2017-06", 42, 9, "base",
     "A classic, strong model based on atrous spatial pyramid pooling.",
     ("Wide receptive field",), dict(batch_size=8)),
    ("smp_linknet", "LinkNet", "Linknet", "2017-07", 32, 6, "small",
     "Additive skip connections; lighter than U-Net.",
     ("Light", "Fast"), dict(batch_size=16)),
    ("smp_pspnet", "PSPNet", "PSPNet", "2016-12", 35, 7, "small",
     "Pyramid scene parsing; aggregates global context.",
     ("Global context",), dict(batch_size=16)),
    ("smp_fpn", "FPN", "FPN", "2016-12", 34, 7, "small",
     "Feature pyramid network; balanced across structures of different sizes.",
     ("Multi-scale", "Balanced"), dict(batch_size=16)),
    ("smp_unetpp", "U-Net++", "UnetPlusPlus", "2018-07", 36, 9, "base",
     "U-Net redesigned with nested skip connections; an advantage on thin structures.",
     ("Thin structures", "Strong on medical imaging"), dict(batch_size=8)),
    ("smp_unet", "U-Net", "Unet", "2015-05", 32, 6, "small",
     "The standard architecture for medical image segmentation; works well even with little data.",
     ("Medical standard", "Good with little data", "Fast"), dict(batch_size=16)),
]

_SEG_SMP = [
    ModelSpec(
        id=_id, display_name=name, task=SEG, backend=SMP, arch=arch, family=name.split()[0],
        released=rel, params_m=pm, default_img_size=512, min_vram_gb=vram, size_class=sz,
        summary=summary, strengths=strengths, needs_encoder=True, weights="imagenet",
        rec=dict(_SEG_DEFAULT, img_size=512, **extra),
    )
    for _id, name, arch, rel, pm, vram, sz, summary, strengths, extra in _SMP_ROWS
]

# The default encoder for smp architectures plus the options highlighted in the UI
DEFAULT_ENCODER = "resnet50"
FEATURED_ENCODERS = [
    ("tu-convnextv2_tiny", "ConvNeXt V2 Tiny (timm) — modern, strong"),
    ("tu-convnextv2_base", "ConvNeXt V2 Base (timm) — strongest, heavy"),
    ("mit_b2", "MiT-B2 — the SegFormer backbone, efficient"),
    ("timm-efficientnet-b4", "EfficientNet-B4 — balanced"),
    ("se_resnext50_32x4d", "SE-ResNeXt50 — strong classic"),
    ("resnet50", "ResNet-50 — universal reference"),
    ("resnet34", "ResNet-34 — fast, low memory"),
    ("timm-efficientnet-b0", "EfficientNet-B0 — lightest"),
]

# ═════════════════════════════════════════════════════════════════════════════
# SEGMENTATION 3D — MONAI
# ═════════════════════════════════════════════════════════════════════════════

_SEG3D_ROWS = [
    ("monai_swinunetr", "Swin UNETR", "SwinUNETR", "2022-01", 62, 16, "large",
     "A 3D U-Net with a Swin transformer encoder; the reference on benchmarks such as BraTS/BTCV.",
     ("3D state of the art", "Transformer encoder"), dict(lr=1e-4, batch_size=2)),
    ("monai_unetr", "UNETR", "UNETR", "2021-03", 92, 18, "large",
     "3D segmentation with a pure ViT encoder; captures long-range context well.",
     ("Long-range context",), dict(lr=1e-4, batch_size=2)),
    ("monai_segresnet", "SegResNet", "SegResNet", "2018-10", 19, 10, "base",
     "The residual 3D network that won BraTS; memory friendly and fast.",
     ("Memory friendly", "Fast"), dict(lr=2e-4, batch_size=2)),
    ("monai_dynunet", "DynUNet (nnU-Net style)", "DynUNet", "2020-12", 31, 12, "base",
     "MONAI's take on nnU-Net; a strong baseline that scales itself to the dataset.",
     ("Self-configuring", "Very strong baseline"), dict(lr=1e-2, optimizer="sgd", batch_size=2)),
    ("monai_attentionunet", "Attention U-Net", "AttentionUnet", "2018-04", 24, 11, "base",
     "A 3D U-Net with attention gates; improves focus on small lesions.",
     ("Small lesions",), dict(lr=2e-4, batch_size=2)),
    ("monai_unet3d", "U-Net 3D", "UNet", "2016-06", 19, 9, "small",
     "The classic 3D U-Net; the most practical option for a quick baseline.",
     ("Simple", "Fast"), dict(lr=2e-4, batch_size=2)),
    ("monai_vnet", "V-Net", "VNet", "2016-06", 45, 12, "base",
     "The classic fully convolutional 3D network, originally proposed together with the Dice loss.",
     ("Classic reference",), dict(lr=2e-4, batch_size=2)),
]

_SEG3D_MONAI = [
    ModelSpec(
        id=_id, display_name=name, task=SEG3D, backend=MONAI, arch=arch, family=name.split()[0],
        released=rel, params_m=pm, default_img_size=96, min_vram_gb=vram, size_class=sz,
        summary=summary, strengths=strengths,
        rec=dict(_SEG_DEFAULT, img_size=96, num_workers=2, **extra),
    )
    for _id, name, arch, rel, pm, vram, sz, summary, strengths, extra in _SEG3D_ROWS
]


# ═════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION — torchvision
# ═════════════════════════════════════════════════════════════════════════════
# The reference implementations, with the weights the published numbers were
# measured on. timm covers more ground; these are here for when a result has to
# match a torchvision baseline exactly, and because they carry no extra
# dependency — torchvision is installed alongside torch either way.

_CLS_TV_ROWS = [
    ("tv_convnext_base", "ConvNeXt Base", "convnext_base", "2022-01", 88.6, 9, "base",
     "The reference ConvNeXt: a CNN retuned with transformer design choices, and still a very strong baseline.",
     ("Strong baseline", "Well understood"), _CNN_MODERN, dict(batch_size=16)),
    ("tv_convnext_tiny", "ConvNeXt Tiny", "convnext_tiny", "2022-01", 28.6, 5, "small",
     "ConvNeXt at a size that trains comfortably on one card.",
     ("Good accuracy/cost", "Fast"), _CNN_MODERN, dict(batch_size=32)),
    ("tv_swin_v2_b", "Swin V2 Base", "swin_v2_b", "2021-11", 87.9, 10, "base",
     "Hierarchical windowed attention; the reference Swin V2 implementation.",
     ("Strong on dense features", "Scales to high resolution"), _VIT_BASE, dict(batch_size=12)),
    ("tv_swin_v2_t", "Swin V2 Tiny", "swin_v2_t", "2021-11", 28.4, 5, "small",
     "The compact Swin V2 — a transformer that still fits a modest GPU.",
     ("Transformer at low cost",), _VIT_SMALL, dict(batch_size=24)),
    ("tv_vit_b_16", "ViT-B/16", "vit_b_16", "2020-10", 86.6, 9, "base",
     "The original Vision Transformer, ImageNet-21k pretrained; the reference point for every ViT result.",
     ("Reference architecture",), _VIT_BASE, dict(batch_size=16, freeze_backbone_epochs=2)),
    ("tv_efficientnet_v2_s", "EfficientNetV2-S", "efficientnet_v2_s", "2021-04", 21.5, 5, "small",
     "Fast to train and accurate for its size; a strong default when the GPU is the constraint.",
     ("Efficient", "Fast to train"), _CNN_MODERN, dict(batch_size=32)),
    ("tv_regnet_y_8gf", "RegNet-Y 8GF", "regnet_y_8gf", "2020-03", 39.4, 6, "small",
     "A design-space-searched CNN; predictable scaling and solid accuracy per FLOP.",
     ("Predictable scaling",), _CNN_CLASSIC, dict(batch_size=24)),
    ("tv_resnet50", "ResNet-50", "resnet50", "2015-12", 25.6, 5, "small",
     "The universal reference. Almost every paper reports it, which makes it the honest first baseline.",
     ("Universal baseline", "Robust", "Fast"), _CNN_CLASSIC, dict(batch_size=32)),
    ("tv_resnet18", "ResNet-18", "resnet18", "2015-12", 11.7, 3, "small",
     "Small and quick — the right choice for a sanity check or a small dataset.",
     ("Very fast", "Low memory"), _CNN_CLASSIC, dict(batch_size=64)),
    ("tv_mobilenet_v3_large", "MobileNetV3 Large", "mobilenet_v3_large", "2019-05", 5.5, 2, "nano",
     "Built for constrained hardware; trains in minutes and exports small.",
     ("Tiny", "Fast inference"), _CNN_CLASSIC, dict(batch_size=64)),
]

_CLS_TV = [
    ModelSpec(
        id=_id, display_name=name, task=CLS, backend=TV, arch=arch,
        family=name.split()[0], released=rel, params_m=pm, default_img_size=224,
        min_vram_gb=vram, size_class=sz, summary=summary, strengths=strengths,
        weights="DEFAULT",
        paper_url="https://pytorch.org/vision/stable/models.html",
        rec=dict(base, **extra),
    )
    for _id, name, arch, rel, pm, vram, sz, summary, strengths, base, extra in _CLS_TV_ROWS
]


# ═════════════════════════════════════════════════════════════════════════════
# SEGMENTATION 2D — torchvision
# ═════════════════════════════════════════════════════════════════════════════
# The encoder arrives ImageNet-pretrained and the head is built fresh for this
# dataset's classes (see trainers/torchvision_common.build_segmenter).

_SEG_TV_ROWS = [
    ("tv_deeplabv3_resnet101", "DeepLabV3 ResNet-101", "deeplabv3_resnet101",
     "2017-06", 61.0, 13, "base",
     "Atrous convolution at several rates, on the deeper ResNet; the strongest of the torchvision heads.",
     ("Multi-scale context", "Strong accuracy"), dict(batch_size=4)),
    ("tv_deeplabv3_resnet50", "DeepLabV3 ResNet-50", "deeplabv3_resnet50",
     "2017-06", 42.0, 10, "base",
     "The standard DeepLabV3 — a dependable segmentation reference.",
     ("Reliable reference", "Multi-scale context"), dict(batch_size=8)),
    ("tv_deeplabv3_mobilenet_v3_large", "DeepLabV3 MobileNetV3", "deeplabv3_mobilenet_v3_large",
     "2017-06", 11.0, 5, "small",
     "DeepLabV3 on a mobile backbone; useful when inference has to be cheap.",
     ("Light", "Fast inference"), dict(batch_size=16)),
    ("tv_fcn_resnet50", "FCN ResNet-50", "fcn_resnet50", "2014-11", 35.3, 9, "small",
     "The architecture that started semantic segmentation; a simple, fast baseline.",
     ("Simple", "Fast"), dict(batch_size=8)),
    ("tv_lraspp_mobilenet_v3_large", "LR-ASPP MobileNetV3", "lraspp_mobilenet_v3_large",
     "2019-05", 3.2, 3, "nano",
     "The lightest option here — built for real-time segmentation on modest hardware.",
     ("Very light", "Real time"), dict(batch_size=16)),
]

_SEG_TV = [
    ModelSpec(
        id=_id, display_name=name, task=SEG, backend=TV, arch=arch,
        family=name.split()[0], released=rel, params_m=pm, default_img_size=512,
        min_vram_gb=vram, size_class=sz, summary=summary, strengths=strengths,
        paper_url="https://pytorch.org/vision/stable/models.html",
        rec=dict(_SEG_DEFAULT, img_size=512, **extra),
    )
    for _id, name, arch, rel, pm, vram, sz, summary, strengths, extra in _SEG_TV_ROWS
]


# ═════════════════════════════════════════════════════════════════════════════
# The catalogue + query helpers
# ═════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY: list[ModelSpec] = sorted(
    _CLS_TIMM + _CLS_TV + _CLS_ULTRA
    + _SEG_ULTRA + _SEG_HF + _SEG_SMP + _SEG_TV + _SEG3D_MONAI,
    key=lambda s: (s.released, s.params_m),
    reverse=True,
)

_BY_ID = {s.id: s for s in MODEL_REGISTRY}


def by_id(spec_id: str) -> ModelSpec:
    if spec_id not in _BY_ID:
        raise KeyError(f"Unknown model id: {spec_id}")
    return _BY_ID[spec_id]


def get(spec_id: str) -> ModelSpec | None:
    return _BY_ID.get(spec_id)


def for_task(task: Task) -> list[ModelSpec]:
    """Models for a task, newest first."""
    return [s for s in MODEL_REGISTRY if s.task == task]


def query(
    task: Task | None = None,
    backends: set[Backend] | None = None,
    size_classes: set[str] | None = None,
    max_vram_gb: float | None = None,
    text: str = "",
) -> list[ModelSpec]:
    """The single entry point behind the catalogue filters."""
    out = MODEL_REGISTRY
    if task is not None:
        out = [s for s in out if s.task == task]
    if backends:
        out = [s for s in out if s.backend in backends]
    if size_classes:
        out = [s for s in out if s.size_class in size_classes]
    if max_vram_gb is not None:
        out = [s for s in out if s.min_vram_gb <= max_vram_gb]
    if text:
        q = text.lower().strip()
        out = [
            s for s in out
            if q in s.display_name.lower() or q in s.family.lower()
            or q in s.arch.lower() or q in s.summary.lower()
            or any(q in t.lower() for t in s.strengths)
        ]
    return out


def backends_for(task: Task) -> list[Backend]:
    seen: list[Backend] = []
    for s in for_task(task):
        if s.backend not in seen:
            seen.append(s.backend)
    return seen


SIZE_CLASS_LABELS = {
    "nano": "Nano (<10M)",
    "small": "Small (10–40M)",
    "base": "Base (40–100M)",
    "large": "Large (>100M)",
}

BACKEND_LABELS = {
    Backend.TIMM: "timm",
    Backend.TORCHVISION: "torchvision",
    Backend.SMP: "smp",
    Backend.HF: "HuggingFace",
    Backend.ULTRALYTICS: "Ultralytics",
    Backend.MONAI: "MONAI",
}
