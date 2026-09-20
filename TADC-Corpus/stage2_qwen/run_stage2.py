# -*- coding: utf-8 -*-
"""TADC-Corpus Stage 2: Qwen-VL semantic labeling."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import cv2
from PIL import Image, ImageDraw, ImageFont

from stage2_qwen.label_converter import (
    build_yolo_lines,
    denormalize_bbox,
    normalize_bbox,
    parse_qwen_response,
    write_raw_qwen_labels,
    write_lines,
)
from stage2_qwen.qwen_client import QwenVLClient
from utils.config import load_config


def read_bbox_json(path: Path) -> List[List[int]]:
    import json
    if not path.is_file():
        raise FileNotFoundError(f"Stage-1 JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    bboxes = []
    for instance in data.get("instances", []):
        bbox = instance.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            bboxes.append([int(float(x)) for x in bbox])
        except (TypeError, ValueError):
            continue
    return bboxes


def resize_for_api(image_bgr, target_long_edge=1024):
    h, w = image_bgr.shape[:2]
    current_long_edge = max(h, w)
    if current_long_edge <= target_long_edge:
        return image_bgr
    scale = target_long_edge / float(current_long_edge)
    return cv2.resize(
        image_bgr,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


def build_prompt(target_classes, normalized_bboxes, class_prompts):
    class_prompts_text = ""
    for cls in target_classes:
        prompts = class_prompts.get(cls, [])
        if prompts:
            class_prompts_text += f"\n类别 {cls} 的提示词:\n" + "\n".join(prompts) + "\n"

    bboxes_desc = "\n".join(
        f"原始框{i + 1}坐标: {bbox}"
        for i, bbox in enumerate(normalized_bboxes)
    )

    return f"""
### 核心任务 ###
分析图片中指定原始框内的内容，仅处理{target_classes}类别的目标。
⚠️ 首要强制要求：原始框内若有2个及以上{target_classes}目标，必须逐个拆分标注，不允许合并、不允许遗漏、不允许框体重叠！

### 基础信息 ###
1. 坐标规则：全图范围被严格归一化为 0~1000 的相对坐标。左上角为原点 [0,0]，右下角为 [1000,1000]。
2. 框坐标格式为 [x0,y0,x1,y1]（必须是 0-1000 范围内的整数）。

### 需要分析的原始框列表 (0~1000归一化坐标) ###
{bboxes_desc}

### 类别提示词参考 ###
{class_prompts_text}

### 处理逻辑 ###
对于每个原始框，请执行以下思维步骤：
1. 观察：框内有多少个独立的个体？（特别是重叠部分，请寻找不同的中心点）
2. 计数：在心里确认数量。
3. 拆分：按照数量，为每一个个体绘制不重叠的、紧贴边缘的千分位坐标框。

### 处理规则 ###
1. 当原始框内包含 ≥2 个目标时，必须拆分，为每个目标单独绘制 [x0,y0,x1,y1] 归一化坐标。
2. 若原始框内没有目标，直接丢弃。
3. 若只有1个目标，微调归一化坐标至完全贴合目标边缘。
4. 仅标注指定的类别。

### 输出格式要求（无任何多余内容，千问必须严格遵守）###
每个目标单独占两行，格式固定为：
类别: 目标类别
目标坐标: [x0,y0,x1,y1]

### 千问专属示例 ###
示例：原始框 [200,300,500,600] 内有1个 person 和1辆 bicycle
正确输出：
类别: person
目标坐标: [210,310,350,580]
类别: bicycle
目标坐标: [360,320,490,590]
""".strip()


def draw_labels(image_path, classes, boxes_xyxy, save_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except Exception:
        font = ImageFont.load_default()

    for class_name, box in zip(classes, boxes_xyxy):
        x0, y0, x1, y1 = map(int, box)
        draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 0, 0), width=3)
        draw.text((x0, max(0, y0 - 30)), class_name, fill=(255, 0, 0), font=font)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)


def run_stage2(config: Dict[str, Any]) -> Path:
    paths = config["paths"]
    cfg = config["stage2"]
    classes = config["dataset"]["classes"]
    class_to_id = {name: idx for idx, name in enumerate(classes)}

    image_dir = Path(paths["image_dir"])
    output_root = Path(paths["output_root"])
    stage1_json_dir = output_root / "stage1" / "json"
    stage2_root = output_root / "stage2"
    raw_dir = stage2_root / "raw_qwen"
    yolo_dir = stage2_root / "yolo_labels"
    vis_dir = stage2_root / "visualization"
    temp_dir = stage2_root / "temp"

    for d in [raw_dir, yolo_dir, temp_dir]:
        d.mkdir(parents=True, exist_ok=True)
    if config["runtime"]["enable_visualization"]:
        vis_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not stage1_json_dir.is_dir():
        raise FileNotFoundError(f"Stage-1 JSON directory not found: {stage1_json_dir}")

    client = QwenVLClient(
        model=str(cfg["model"]),
        max_retries=int(cfg["max_retries"]),
        retry_interval=float(cfg["retry_interval"]),
    )

    image_files = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    max_workers = int(cfg["max_workers"])
    target_long_edge = int(cfg["target_long_edge"])
    class_prompts = cfg.get("class_prompts", {})

    print("=" * 72)
    print("TADC-Corpus | Stage 2: Qwen-VL Semantic Labeling")
    print("=" * 72)
    print(f"IMAGE_DIR       : {image_dir}")
    print(f"STAGE1_JSON_DIR : {stage1_json_dir}")
    print(f"YOLO_OUTPUT     : {yolo_dir}")
    print(f"MODEL           : {cfg['model']}")
    print(f"MAX_WORKERS     : {max_workers}")

    def process_single_image(img_path: Path) -> str:
        img_name = img_path.name
        stem = img_path.stem
        raw_save_path = raw_dir / f"{stem}.txt"
        yolo_save_path = yolo_dir / f"{stem}.txt"

        if yolo_save_path.exists():
            return f"[SKIP] {img_name} - output already exists"

        bbox_file = stage1_json_dir / f"{stem}.bbox.json"
        if not bbox_file.exists():
            return f"[WARN] {img_name} - Stage-1 JSON not found"

        original_bboxes = read_bbox_json(bbox_file)
        if not original_bboxes:
            write_raw_qwen_labels(raw_save_path, [], [])
            write_lines(yolo_save_path, [])
            return f"[WARN] {img_name} - no valid Stage-1 boxes"

        img_cv = cv2.imread(str(img_path))
        if img_cv is None:
            return f"[WARN] {img_name} - image read failed"

        original_h, original_w = img_cv.shape[:2]
        resized_img = resize_for_api(img_cv, target_long_edge)
        temp_img_path = temp_dir / img_name
        cv2.imwrite(str(temp_img_path), resized_img)
        api_img_path = "file://" + str(temp_img_path.resolve())

        normalized_bboxes = [
            normalize_bbox(bbox, original_w, original_h)
            for bbox in original_bboxes
        ]

        prompt = build_prompt(
            classes,
            normalized_bboxes,
            class_prompts,
        )
        print(f"[PROCESS] {img_name}")
        response_text = client.call(api_img_path, prompt)
        if response_text is None:
            return f"[ERROR] {img_name} - Qwen API failed"

        pred_classes, pred_norm_bboxes = parse_qwen_response(response_text)
        if not pred_classes or not pred_norm_bboxes:
            write_raw_qwen_labels(raw_save_path, [], [])
            write_lines(yolo_save_path, [])
            return f"[WARN] {img_name} - no valid Qwen labels"

        restored_boxes = [
            denormalize_bbox(bbox, original_w, original_h)
            for bbox in pred_norm_bboxes
        ]

        write_raw_qwen_labels(
            raw_save_path,
            pred_classes,
            restored_boxes,
        )

        yolo_lines, warnings = build_yolo_lines(
            classes=pred_classes,
            boxes_0_1000=pred_norm_bboxes,
            class_to_id=class_to_id,
            image_w=original_w,
            image_h=original_h,
        )
        write_lines(yolo_save_path, yolo_lines)

        if config["runtime"]["enable_visualization"]:
            try:
                draw_labels(
                    img_path,
                    pred_classes,
                    restored_boxes,
                    vis_dir / img_name,
                )
            except Exception as exc:
                print(f"[WARN] {img_name} - visualization failed: {exc}")

        if warnings:
            return f"[SUCCESS] {img_name} - {len(yolo_lines)} labels; {len(warnings)} warning(s)"
        return f"[SUCCESS] {img_name} - {len(yolo_lines)} labels"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_image, img): img
            for img in image_files
        }
        for future in as_completed(futures):
            img_path = futures[future]
            try:
                print(future.result())
            except Exception as exc:
                print(f"[CRITICAL] {img_path.name}: {exc}")

    print(f">>> Stage 2 complete: {yolo_dir}")
    return yolo_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="TADC-Corpus Stage 2")
    parser.add_argument("--config", required=True, help="Path to pipeline.yaml")
    args = parser.parse_args()
    run_stage2(load_config(args.config))


if __name__ == "__main__":
    main()
