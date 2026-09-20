# -*- coding: utf-8 -*-
"""TADC-Corpus Stage 1: DINOv3-based class-agnostic object discovery."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from stage1_dinov3.attnwalk import AttnWalkSegmenter
from stage1_dinov3.dino_attention import (
    apply_rope_to_qk,
    clear_runtime_storage,
    hijack_n_blocks_attention,
    runtime_storage,
)
from utils.config import load_config


def get_slice_bboxes(image_w: int, image_h: int, overlap_ratio: float = 0.2, crop_ratio: float = 0.6, min_crop_size: int = 100) -> List[Tuple[int, int, int, int]]:
    w_step = int(image_w * (1 - overlap_ratio))
    h_step = int(image_h * (1 - overlap_ratio))
    w_step = max(w_step, 1)
    h_step = max(h_step, 1)
    crop_w = int(image_w * crop_ratio)
    crop_h = int(image_h * crop_ratio)
    slice_bboxes = [(0, 0, image_w, image_h)]

    for y in range(0, image_h, h_step):
        y_max = image_h
        for x in range(0, image_w, w_step):
            x_max = min(x + crop_w, image_w)
            y_max = min(y + crop_h, image_h)
            if x_max - x < min_crop_size or y_max - y < min_crop_size:
                continue
            if x == 0 and y == 0 and x_max == image_w and y_max == image_h:
                continue
            slice_bboxes.append((x, y, x_max, y_max))
            if x_max == image_w:
                break
        if y_max == image_h:
            break
    return slice_bboxes


def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
    bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union if union > 0 else 0.0


def apply_nms_to_instances(instances, iou_thresh=0.45):
    instances = sorted(instances, key=lambda x: x["area"], reverse=True)
    keep = []
    for inst in instances:
        if not any(compute_iou(inst["bbox"], k["bbox"]) > iou_thresh for k in keep):
            keep.append(inst)
    return keep


def normalize_to_uint8_colormap(canvas, original_h, original_w, colormap):
    if canvas.max() == canvas.min():
        return np.zeros((original_h, original_w, 3), dtype=np.uint8)
    norm = (canvas - canvas.min()) / (canvas.max() - canvas.min() + 1e-8)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), colormap)


def resize_for_display(img, text_label, vis_w=512, vis_h=512):
    resized = cv2.resize(img, (vis_w, vis_h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(resized, (0, 0), (200, 30), (0, 0, 0), -1)
    cv2.putText(resized, text_label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return resized


def save_visualization(img_pil, final_instances, global_mask_canvas, global_attn_canvas, global_pca_canvas, global_refined_canvas, save_path, vis_size=512):
    orig_w, orig_h = img_pil.size
    attn_color = normalize_to_uint8_colormap(global_attn_canvas, orig_h, orig_w, cv2.COLORMAP_INFERNO)
    pca_color = normalize_to_uint8_colormap(global_pca_canvas, orig_h, orig_w, cv2.COLORMAP_BONE)
    refined_color = normalize_to_uint8_colormap(global_refined_canvas, orig_h, orig_w, cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    color_mask = np.zeros_like(img_bgr)
    color_mask[global_mask_canvas == 255] = [0, 0, 255]
    overlay_vis = cv2.addWeighted(img_bgr, 0.8, color_mask, 0.4, 0)

    for inst in final_instances:
        x_min, y_min, w, h = inst["bbox"]
        cv2.rectangle(overlay_vis, (x_min, y_min), (x_min + w, y_min + h), (0, 255, 0), 2)
        cv2.putText(overlay_vis, f"Obj{inst['id']}", (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    panels = [
        resize_for_display(overlay_vis, "1. Final BBox & Mask", vis_size, vis_size),
        resize_for_display(attn_color, "2. Attention Map", vis_size, vis_size),
        resize_for_display(pca_color, "3. PCA Saliency", vis_size, vis_size),
        resize_for_display(refined_color, "4. Graph Diffused", vis_size, vis_size),
        resize_for_display(cv2.cvtColor(global_mask_canvas, cv2.COLOR_GRAY2BGR), "5. Binary Mask", vis_size, vis_size),
    ]
    final_multi_view = np.hstack(panels)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), final_multi_view)


def resolve_device(value: str) -> torch.device:
    value = str(value).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_dinov3_model(config: Dict[str, Any], device: torch.device):
    repo = config["paths"]["dinov3_repo"]
    checkpoint = config["paths"]["dinov3_checkpoint"]
    stage1_cfg = config["stage1"]
    model_name = stage1_cfg["model_name"]
    n_layers = int(stage1_cfg["n_layers_fusion"])

    if not os.path.isdir(repo):
        raise FileNotFoundError(f"DINOv3 repository not found: {repo}")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"DINOv3 checkpoint not found: {checkpoint}")

    print(">>> Loading DINOv3 model...")
    model = torch.hub.load(repo, model_name, source="local", weights=checkpoint)
    model = model.eval().to(device)
    n_storage = model.n_storage_tokens
    hijack_n_blocks_attention(model, n=n_layers)
    return model, n_storage


@torch.no_grad()
def run_stage1(config: Dict[str, Any]) -> Path:
    paths = config["paths"]
    cfg = config["stage1"]
    image_dir = Path(paths["image_dir"])
    stage1_root = Path(paths["output_root"]) / "stage1"
    json_dir = stage1_root / "json"
    vis_dir = stage1_root / "visualization"
    json_dir.mkdir(parents=True, exist_ok=True)
    if config["runtime"]["enable_visualization"]:
        vis_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    device = resolve_device(config["runtime"]["device"])
    print("=" * 72)
    print("TADC-Corpus | Stage 1: DINOv3 Object Discovery")
    print("=" * 72)
    print(f"IMAGE_DIR       : {image_dir}")
    print(f"OUTPUT_ROOT     : {stage1_root}")
    print(f"DEVICE          : {device}")
    print(f"DINOv3 MODEL    : {cfg['model_name']}")
    print(f"LAYERS FUSION   : {cfg['n_layers_fusion']}")
    print(f"SLICING ENABLED : {cfg['slicing_enabled']}")

    model, n_storage = load_dinov3_model(config, device)
    input_size = int(cfg["input_size"])
    transform = T.Compose([
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    segmenter = AttnWalkSegmenter(
        k_neighbors=int(cfg["k_neighbors"]),
        beta=float(cfg["beta"]),
        max_iter=int(cfg["max_iter"]),
        device=str(device),
    )
    vis_h = vis_w = input_size
    image_files = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f">>> Processing {len(image_files)} images...")

    for img_path in tqdm(image_files):
        try:
            img_pil = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img_pil.size
        except Exception as exc:
            print(f"[WARN] Failed to read {img_path.name}: {exc}")
            continue

        filename_stem = img_path.stem
        all_instances = []
        global_mask_canvas = np.zeros((orig_h, orig_w), dtype=np.uint8)
        global_attn_canvas = np.zeros((orig_h, orig_w), dtype=np.float32)
        global_pca_canvas = np.zeros((orig_h, orig_w), dtype=np.float32)
        global_refined_canvas = np.zeros((orig_h, orig_w), dtype=np.float32)

        if cfg["slicing_enabled"]:
            crop_boxes = get_slice_bboxes(
                orig_w,
                orig_h,
                overlap_ratio=float(cfg["overlap_ratio"]),
                crop_ratio=float(cfg["crop_ratio"]),
                min_crop_size=int(cfg["min_crop_size"]),
            )
        else:
            crop_boxes = [(0, 0, orig_w, orig_h)]

        for cx_min, cy_min, cx_max, cy_max in crop_boxes:
            img_crop = img_pil.crop((cx_min, cy_min, cx_max, cy_max))
            crop_w, crop_h = img_crop.size
            x = transform(img_crop).unsqueeze(0).to(device)

            layers = model.get_intermediate_layers(
                x,
                n=int(cfg["n_layers_fusion"]),
                reshape=True,
                norm=True,
            )
            feat_stack = torch.stack(layers)
            spatial_energy = torch.norm(feat_stack, dim=2)
            N_layers, B, H, W_grid = spatial_energy.shape
            flat_energy = spatial_energy.reshape(N_layers, B, -1)
            probs = flat_energy / (flat_energy.sum(dim=-1, keepdim=True) + 1e-10)
            entropies = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            weights = F.softmax(-entropies * float(cfg["softmax_scale"]), dim=0)
            weights_expanded = weights.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            feat_hw = (feat_stack * weights_expanded).sum(dim=0)
            b, c, h_g, w_g = feat_hw.shape
            feat_flat = feat_hw[0].permute(1, 2, 0).reshape(h_g * w_g, c).cpu().numpy()

            attn_list = []
            total_blocks = len(model.blocks)
            for layer_idx in range(total_blocks - int(cfg["n_layers_fusion"]), total_blocks):
                if layer_idx not in runtime_storage:
                    continue
                layer_data = runtime_storage[layer_idx]
                x_input = layer_data.get("x")
                rope_data = layer_data.get("rope")
                if x_input is None:
                    continue
                attn_layer = model.blocks[layer_idx].attn
                B_attn, N_attn, C_in = x_input.shape
                num_heads = attn_layer.num_heads
                head_dim = C_in // num_heads
                qkv = attn_layer.qkv(x_input).reshape(B_attn, N_attn, 3, num_heads, head_dim)
                q, k, v = qkv.unbind(2)
                q, k = q.transpose(1, 2), k.transpose(1, 2)
                if rope_data is not None:
                    q, k = apply_rope_to_qk(q, k, rope_data)
                scale = head_dim ** -0.5
                attn = (q @ k.transpose(-2, -1)) * scale
                attn = attn.softmax(dim=-1)
                start_idx = 1 + n_storage
                attn_list.append(attn[0, :, 0, start_idx:].mean(dim=0))

            attn_for_seg = None
            if attn_list:
                attn_stack = torch.stack(attn_list)
                probs = attn_stack / (attn_stack.sum(dim=1, keepdim=True) + 1e-10)
                entropies = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                weights = F.softmax(-entropies * float(cfg["softmax_scale"]), dim=0)
                attn_fused = (attn_stack * weights.unsqueeze(1)).sum(dim=0)
                if attn_fused.shape[-1] == h_g * w_g:
                    attn_for_seg = attn_fused

            mask, refined_score, pca_map, cls_attn = segmenter.segment(
                feat_flat,
                attn_for_seg,
                (h_g, w_g),
            )

            local_mask_real = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
            global_mask_canvas[cy_min:cy_max, cx_min:cx_max] = cv2.bitwise_or(
                global_mask_canvas[cy_min:cy_max, cx_min:cx_max],
                local_mask_real,
            )

            attn_2d = cls_attn.reshape(h_g, w_g).cpu().numpy()
            attn_real = cv2.resize(attn_2d, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
            global_attn_canvas[cy_min:cy_max, cx_min:cx_max] = np.maximum(
                global_attn_canvas[cy_min:cy_max, cx_min:cx_max],
                attn_real,
            )

            pca_2d = pca_map.reshape(h_g, w_g).cpu().numpy()
            pca_real = cv2.resize(pca_2d, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
            global_pca_canvas[cy_min:cy_max, cx_min:cx_max] = np.maximum(
                global_pca_canvas[cy_min:cy_max, cx_min:cx_max],
                pca_real,
            )

            refined_real = cv2.resize(refined_score, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
            global_refined_canvas[cy_min:cy_max, cx_min:cx_max] = np.maximum(
                global_refined_canvas[cy_min:cy_max, cx_min:cx_max],
                refined_real,
            )

            mask_resized = cv2.resize(mask, (vis_w, vis_h), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < float(cfg["bbox_area_threshold"]):
                    continue
                x_box, y_box, w_box, h_box = cv2.boundingRect(cnt)
                abs_x_min = x_box * (crop_w / vis_w) + cx_min
                abs_y_min = y_box * (crop_h / vis_h) + cy_min
                abs_x_max = (x_box + w_box) * (crop_w / vis_w) + cx_min
                abs_y_max = (y_box + h_box) * (crop_h / vis_h) + cy_min
                abs_x_min = max(0, abs_x_min)
                abs_y_min = max(0, abs_y_min)
                abs_x_max = min(orig_w, abs_x_max)
                abs_y_max = min(orig_h, abs_y_max)
                abs_w = abs_x_max - abs_x_min
                abs_h = abs_y_max - abs_y_min
                if abs_w <= 0 or abs_h <= 0:
                    continue
                real_area = abs_w * abs_h
                center_x = abs_x_min + abs_w / 2.0
                center_y = abs_y_min + abs_h / 2.0
                bbox_norm = [abs_x_min / orig_w, abs_y_min / orig_h, abs_w / orig_w, abs_h / orig_h]
                bbox_xyxy_norm = [abs_x_min / orig_w, abs_y_min / orig_h, abs_x_max / orig_w, abs_y_max / orig_h]
                all_instances.append({
                    "bbox": [int(abs_x_min), int(abs_y_min), int(abs_w), int(abs_h)],
                    "area": int(real_area),
                    "centroid": [center_x, center_y],
                    "cluster_id": 0,
                    "bbox_xyxy": [int(abs_x_min), int(abs_y_min), int(abs_x_max), int(abs_y_max)],
                    "bbox_normalized": bbox_norm,
                    "bbox_xyxy_normalized": bbox_xyxy_norm,
                })

            clear_runtime_storage()

        final_instances = apply_nms_to_instances(
            all_instances,
            iou_thresh=float(cfg["nms_iou_threshold"]),
        )
        for i, inst in enumerate(final_instances):
            inst["id"] = i + 1

        json_data = {
            "image_name": filename_stem,
            "image_size": [orig_w, orig_h],
            "num_instances": len(final_instances),
            "instances": final_instances,
        }
        json_path = json_dir / f"{filename_stem}.bbox.json"
        with json_path.open("w", encoding="utf-8") as f_out:
            json.dump(json_data, f_out, indent=2, ensure_ascii=False)

        if config["runtime"]["enable_visualization"]:
            save_visualization(
                img_pil,
                final_instances,
                global_mask_canvas,
                global_attn_canvas,
                global_pca_canvas,
                global_refined_canvas,
                vis_dir / f"{filename_stem}_pipeline.png",
                vis_size=input_size,
            )

    print(f">>> Stage 1 complete: {json_dir}")
    return json_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="TADC-Corpus Stage 1")
    parser.add_argument("--config", required=True, help="Path to pipeline.yaml")
    args = parser.parse_args()
    run_stage1(load_config(args.config))


if __name__ == "__main__":
    main()
