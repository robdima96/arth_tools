# -*- coding: utf-8 -*-
"""YOLOv3-Tiny (1 class). Architecture follows Gu et al. 2022 Fig. 2; implemented here (MIT)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

N_ANCHORS = 3
N_OUT = 5 + 1  # xywh + obj + 1 class


class _ConvBNLeaky(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1) -> None:
        super().__init__()
        pad = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class YoloV3Tiny(nn.Module):
    """Two-scale YOLOv3-Tiny backbone + heads for a single class."""

    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        out = N_ANCHORS * (5 + self.num_classes)
        self.conv1 = _ConvBNLeaky(3, 16)
        self.conv2 = _ConvBNLeaky(16, 32)
        self.conv3 = _ConvBNLeaky(32, 64)
        self.conv4 = _ConvBNLeaky(64, 128)
        self.conv5 = _ConvBNLeaky(128, 256)
        self.conv6 = _ConvBNLeaky(256, 512)
        self.conv7 = _ConvBNLeaky(512, 1024)
        self.conv8 = _ConvBNLeaky(1024, 256, k=1)
        self.conv9 = _ConvBNLeaky(256, 512)
        self.head_large = nn.Conv2d(512, out, 1)
        self.conv_route = _ConvBNLeaky(256, 128, k=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_fuse = _ConvBNLeaky(128 + 256, 256)
        self.head_small = nn.Conv2d(256, out, 1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        route = self.conv5(x)
        x = self.pool(route)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        large = self.head_large(self.conv9(x))
        up = self.upsample(self.conv_route(x))
        fused = torch.cat([up, route], dim=1)
        small = self.head_small(self.conv_fuse(fused))
        return large, small


def _mesh(h: int, w: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    gy, gx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    return gx.float(), gy.float()


def decode_heads(
    heads: Sequence[torch.Tensor],
    *,
    img_size: int,
    anchors: Sequence[float],
) -> torch.Tensor:
    """Decode YOLO heads to (B, N, 6) = x1,y1,x2,y2,obj,cls in pixel xyxy."""
    anchor_pairs = [(float(anchors[i]), float(anchors[i + 1])) for i in range(0, len(anchors), 2)]
    small_anchors = anchor_pairs[:3]
    large_anchors = anchor_pairs[3:6] if len(anchor_pairs) >= 6 else anchor_pairs[:3]
    decoded: list[torch.Tensor] = []
    for pred, anch in zip(heads, (large_anchors, small_anchors)):
        b, ch, h, w = pred.shape
        n_a = 3
        pred = pred.view(b, n_a, ch // n_a, h, w)
        gx, gy = _mesh(h, w, pred.device)
        stride = float(img_size) / float(h)
        xy_raw = torch.sigmoid(pred[:, :, 0:2])
        px = (xy_raw[:, :, 0] + gx) * stride
        py = (xy_raw[:, :, 1] + gy) * stride
        aw = torch.tensor([a[0] for a in anch], device=pred.device, dtype=pred.dtype).view(1, n_a, 1, 1)
        ah = torch.tensor([a[1] for a in anch], device=pred.device, dtype=pred.dtype).view(1, n_a, 1, 1)
        wh_raw = torch.exp(pred[:, :, 2:4].clamp(-8, 8))
        pw = wh_raw[:, :, 0] * aw
        ph = wh_raw[:, :, 1] * ah
        obj = torch.sigmoid(pred[:, :, 4:5])
        cls = torch.sigmoid(pred[:, :, 5:6]) if pred.size(2) > 5 else obj
        x1 = px - pw / 2
        y1 = py - ph / 2
        x2 = px + pw / 2
        y2 = py + ph / 2
        boxes = torch.stack([x1, y1, x2, y2], dim=2)
        boxes = torch.cat([boxes, obj, cls], dim=2)
        decoded.append(boxes.permute(0, 1, 3, 4, 2).reshape(b, -1, 6))
    return torch.cat(decoded, dim=1)


def yolo_loss(
    heads: Sequence[torch.Tensor],
    targets: torch.Tensor,
    *,
    img_size: int,
    anchors: Sequence[float],
) -> torch.Tensor:
    """targets: (B, 5) = cls, xc, yc, w, h normalized. One box per image."""
    pred_boxes = decode_heads(heads, img_size=img_size, anchors=anchors)
    device = pred_boxes.device
    b = pred_boxes.size(0)
    loss = torch.zeros((), device=device)
    for i in range(b):
        gt = targets[i]
        gx = float(gt[1]) * img_size
        gy = float(gt[2]) * img_size
        gw = float(gt[3]) * img_size
        gh = float(gt[4]) * img_size
        g = torch.tensor([gx - gw / 2, gy - gh / 2, gx + gw / 2, gy + gh / 2], device=device)
        boxes = pred_boxes[i]
        ious = _box_iou(boxes[:, :4], g.unsqueeze(0)).squeeze(1)
        best = int(torch.argmax(ious).item())
        pred = boxes[best]
        loss = loss + torch.nn.functional.smooth_l1_loss(pred[:4], g)
        loss = loss + torch.nn.functional.binary_cross_entropy(pred[4].clamp(1e-4, 1 - 1e-4), torch.ones((), device=device))
        obj = boxes[:, 4]
        neg = torch.ones_like(obj)
        neg[best] = 0.0
        if int(neg.sum().item()) > 0:
            loss = loss + 0.1 * torch.nn.functional.binary_cross_entropy(
                obj.clamp(1e-4, 1 - 1e-4),
                torch.zeros_like(obj),
                weight=neg,
            )
    return loss / max(b, 1)


def _box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a (N,4) xyxy, b (M,4) xyxy → (N, M)."""
    tl = torch.maximum(a[:, None, :2], b[None, :, :2])
    br = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-8)
