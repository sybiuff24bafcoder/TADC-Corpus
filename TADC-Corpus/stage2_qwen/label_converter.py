# -*- coding: utf-8 -*-
"""Qwen coordinate parsing and conversion to standard YOLO labels."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def normalize_bbox(bbox_xyxy: Sequence[float], image_w: int, image_h: int) -> List[int]:
    return [
        max(0, min(1000, int((bbox_xyxy[0] / image_w) * 1000))),
        max(0, min(1000, int((bbox_xyxy[1] / image_h) * 1000))),
        max(0, min(1000, int((bbox_xyxy[2] / image_w) * 1000))),
        max(0, min(1000, int((bbox_xyxy[3] / image_h) * 1000))),
    ]


def denormalize_bbox(bbox: Sequence[float], image_w: int, image_h: int) -> List[int]:
    return [
        int((bbox[0] / 1000.0) * image_w),
        int((bbox[1] / 1000.0) * image_h),
        int((bbox[2] / 1000.0) * image_w),
        int((bbox[3] / 1000.0) * image_h),
    ]


def parse_qwen_response(response_text: str | None) -> Tuple[List[str], List[List[int]]]:
    if response_text is None:
        raise ValueError("Qwen response is None.")

    classes: List[str] = []
    boxes: List[List[int]] = []
    pattern = re.compile(
        r"类别\s*:\s*(.+?)\s*[\r\n]+\s*目标坐标\s*:\s*(\[[^\]]+\])",
        re.MULTILINE,
    )

    for match in pattern.finditer(response_text):
        class_name = match.group(1).replace("*", "").replace("'", "").strip()
        try:
            bbox = json.loads(match.group(2))
        except json.JSONDecodeError:
            continue
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            bbox = [int(float(v)) for v in bbox]
        except (TypeError, ValueError):
            continue
        if not all(0 <= v <= 1000 for v in bbox):
            continue
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        classes.append(class_name)
        boxes.append(bbox)

    return classes, boxes


def pixel_xyxy_to_yolo(bbox_xyxy: Sequence[float], image_w: int, image_h: int) -> List[float]:
    x0, y0, x1, y1 = map(float, bbox_xyxy)
    x0 = max(0.0, min(float(image_w), x0))
    y0 = max(0.0, min(float(image_h), y0))
    x1 = max(0.0, min(float(image_w), x1))
    y1 = max(0.0, min(float(image_h), y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid bbox: {bbox_xyxy}")
    return [
        ((x0 + x1) / 2.0) / image_w,
        ((y0 + y1) / 2.0) / image_h,
        (x1 - x0) / image_w,
        (y1 - y0) / image_h,
    ]


def build_yolo_lines(
    classes: Sequence[str],
    boxes_0_1000: Sequence[Sequence[int]],
    class_to_id: Dict[str, int],
    image_w: int,
    image_h: int,
) -> Tuple[List[str], List[str]]:
    lines: List[str] = []
    warnings: List[str] = []

    for class_name, box_norm in zip(classes, boxes_0_1000):
        class_name = class_name.strip()
        if class_name not in class_to_id:
            warnings.append(f"Unknown class returned by Qwen: {class_name}")
            continue
        bbox_xyxy = denormalize_bbox(box_norm, image_w, image_h)
        try:
            cx, cy, w, h = pixel_xyxy_to_yolo(bbox_xyxy, image_w, image_h)
        except ValueError as exc:
            warnings.append(str(exc))
            continue
        lines.append(
            f"{class_to_id[class_name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        )

    return lines, warnings


def write_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def write_raw_qwen_labels(path: Path, classes: Sequence[str], boxes_xyxy: Sequence[Sequence[int]]) -> None:
    lines = [
        f"{cls} {int(box[0])} {int(box[1])} {int(box[2])} {int(box[3])}"
        for cls, box in zip(classes, boxes_xyxy)
    ]
    write_lines(path, lines)
