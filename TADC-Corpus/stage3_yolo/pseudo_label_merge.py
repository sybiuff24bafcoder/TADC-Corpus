# -*- coding: utf-8 -*-
"""Adaptive pseudo-label evolution fusion used by Stage 3."""

import os
import json
import numpy as np
from pathlib import Path
from glob import glob
from typing import List, Optional, Tuple, Set, Dict

# 定义标签类型: (cls_id, x, y, w, h, conf)
Label = Tuple[int, float, float, float, float, float]


# ---------------------------------------------------------------------------
# Label conversion and pseudo-label fusion
# ---------------------------------------------------------------------------
# ==========================================
def labels5_to_labels6(src_dir: Path, dst_dir: Path, default_conf: float = 1.0) -> None:
    """把没有置信度的 YOLO 标签(5 列)转换为带置信度(6 列)。"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not src_dir.exists():
        print(f"[Warning] Source dir {src_dir} does not exist.")
        return
    for txt in src_dir.glob("*.txt"):
        out_txt = dst_dir / txt.name
        with txt.open("r", encoding="utf-8") as f_in, out_txt.open("w", encoding="utf-8") as f_out:
            for line in f_in:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) < 5: continue
                cls_id = parts[0]
                try:
                    x, y, w, h = map(float, parts[1:5])
                except ValueError:
                    continue
                f_out.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {default_conf:.6f}\n")


def labels6_to_labels5(src_dir: Path, dst_dir: Path) -> None:
    """把带置信度的标签(6 列)转为 YOLO 训练使用的 5 列格式。"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not src_dir.exists():
        print(f"[Warning] Source dir {src_dir} does not exist.")
        return
    for txt in src_dir.glob("*.txt"):
        out_txt = dst_dir / txt.name
        with txt.open("r", encoding="utf-8") as f_in, out_txt.open("w", encoding="utf-8") as f_out:
            for line in f_in:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) < 5: continue
                f_out.write(" ".join(parts[:5]) + "\n")


# ==========================================
# 2. 标签解析与 IoU 计算模块
# ==========================================
def _parse_label_line(line: str, default_conf: float = 1.0) -> Optional[Label]:
    line = line.strip()
    if not line: return None
    parts = line.split()
    if len(parts) < 5: return None
    try:
        cls_id = int(float(parts[0]))
    except ValueError:
        return None
    nums = []
    for p in parts[1:]:
        try:
            nums.append(float(p))
        except ValueError:
            return None
    if len(nums) == 4:
        x, y, w, h = nums
        conf = default_conf
    elif len(nums) == 5:
        x1, y1, w1, h1, c1 = nums
        if 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= w1 <= 1.0 and 0.0 <= h1 <= 1.0 and 0.0 <= c1 <= 1.0:
            x, y, w, h, conf = x1, y1, w1, h1, c1
        else:
            c2, x2, y2, w2, h2 = nums
            x, y, w, h, conf = x2, y2, w2, h2, c2
    else:
        return None
    return cls_id, x, y, w, h, conf


def _load_labels(path: str, default_conf: float = 1.0) -> List[Label]:
    if not os.path.exists(path): return []
    labels: List[Label] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = _parse_label_line(line, default_conf)
            if parsed is not None: labels.append(parsed)
    return labels


def _save_labels(path: str, labels: List[Label]) -> None:
    """【修改】永远生成标签文件，即使是空的，不删除任何文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for cls_id, x, y, w, h, conf in labels:
            f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")


def _xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    return x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0


def _box_iou_xywh(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    ax0, ay0, ax1, ay1 = _xywh_to_xyxy(x1, y1, w1, h1)
    bx0, by0, bx1, by1 = _xywh_to_xyxy(x2, y2, w2, h2)
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    iw = max(0.0, inter_x1 - inter_x0)
    ih = max(0.0, inter_y1 - inter_y0)
    inter = iw * ih
    if inter <= 0.0: return 0.0
    area1 = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area2 = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area1 + area2 - inter
    if union <= 0.0: return 0.0
    return inter / union


# ==========================================
# 3. 宏观统计与微观融合模块
# ==========================================
def _get_dynamic_threshold_and_mean(confs: List[float], bins: int = 256) -> Tuple[float, float]:
    if not confs:
        return 0.40, 0.70
    if max(confs) - min(confs) < 1e-3 or len(confs) < 5:
        val = confs[0] if confs else 0.5
        return max(0.20, val * 0.8), val

    hist, bin_edges = np.histogram(confs, bins=bins, range=(0.0, 1.0))
    hist_prob = hist / hist.sum()
    omega = np.cumsum(hist_prob)
    mu = np.cumsum(hist_prob * np.linspace(0.0, 1.0, bins))
    mu_t = mu[-1]

    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_b_sq = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega))
    sigma_b_sq = np.nan_to_num(sigma_b_sq, nan=0.0, posinf=0.0, neginf=0.0)

    max_idx = np.argmax(sigma_b_sq)
    tau_dynamic = (bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2.0
    tau_dynamic = max(0.30, min(0.75, tau_dynamic))

    fg_confs = [c for c in confs if c > tau_dynamic]
    mu_fg = np.mean(fg_confs) if fg_confs else tau_dynamic + 0.1

    return float(tau_dynamic), float(mu_fg)


def calculate_global_class_thresholds(new_dir: str) -> dict:
    print("[Otsu Stats] Collecting global confidence distributions for Class-aware Otsu...")
    class_confs = {}
    for txt_path in glob(os.path.join(new_dir, "*.txt")):
        labels = _load_labels(txt_path)
        for lb in labels:
            cls_id, conf = lb[0], lb[5]
            if cls_id not in class_confs:
                class_confs[cls_id] = []
            class_confs[cls_id].append(conf)

    class_thresholds = {}
    print("[Otsu Stats] Computed global thresholds:")
    for cls_id, confs in class_confs.items():
        tau, mu = _get_dynamic_threshold_and_mean(confs)
        class_thresholds[cls_id] = (tau, mu)
        print(f"  - Class {cls_id}: Base Tau={tau:.3f}, Mean={mu:.3f} (基于 {len(confs)} 个候选框)")
    return class_thresholds


def merge_image_labels(old_labels: List[Label], new_labels: List[Label], round_idx: int, class_thresholds: dict,
                       iou_thr: float = 0.5) -> List[Label]:
    merged: List[Label] = []

    used_old = [False] * len(old_labels)
    used_new = [False] * len(new_labels)

    classes = sorted({lb[0] for lb in old_labels} | {lb[0] for lb in new_labels})

    for cls_id in classes:
        old_idx = [i for i, lb in enumerate(old_labels) if lb[0] == cls_id]
        new_idx = [j for j, lb in enumerate(new_labels) if lb[0] == cls_id]

        # 获取全局公平的生存/准入门槛
        base_tau, mu_fg = class_thresholds.get(cls_id, (0.45, 0.70))

        # 正常类别：常规淘汰
        tau_dynamic = base_tau
        decay_rate = 0.80

        # --- 法则 1：同类别空间匹配成功 ---
        if old_idx and new_idx:
            pairs = []
            for i in old_idx:
                for j in new_idx:
                    iou = _box_iou_xywh(old_labels[i][1:5], new_labels[j][1:5])
                    if iou >= iou_thr:
                        pairs.append((iou, i, j))
            pairs.sort(key=lambda x: x[0], reverse=True)

            for iou, i, j in pairs:
                if used_old[i] or used_new[j]: continue

                # 完美实体绑定逻辑（优胜劣汰，绝不造缝合怪）
                raw_old_conf = old_labels[i][5]
                new_conf = new_labels[j][5]

                # 第一轮：剥夺旧框虚假高分，用 mu_fg 还原真实身价
                if round_idx == 1:
                    actual_old_conf = mu_fg
                else:
                    actual_old_conf = raw_old_conf

                # 用真实的比较值进行对比，优胜劣汰
                if new_conf >= actual_old_conf:
                    # 新框更自信：完整保留新框实体
                    selected_box = new_labels[j]
                else:
                    # 旧框更可靠：完整保留旧框实体
                    if round_idx == 1:
                        # 第一轮：必须保留旧框坐标，并将虚高分数修正为 mu_fg
                        cls_id_box, x_box, y_box, w_box, h_box, _ = old_labels[i]
                        selected_box = (cls_id_box, x_box, y_box, w_box, h_box, mu_fg)
                    else:
                        selected_box = old_labels[i]

                merged.append(selected_box)
                used_old[i] = True
                used_new[j] = True

        # --- 法则 3：未匹配的历史标签 (时序衰减与淘汰) ---
        for i in old_idx:
            if not used_old[i]:
                lb_cls_id, x, y, w, h, raw_old_conf = old_labels[i]

                # 【修改】第一轮所有历史标签都必须修正为mu_fg，不管是否匹配
                if round_idx == 1:
                    base_conf = mu_fg
                else:
                    base_conf = raw_old_conf

                # 使用动态的衰减率
                decayed_conf = base_conf * decay_rate

                # 历史记忆考核生存
                if decayed_conf >= tau_dynamic:
                    merged.append((lb_cls_id, x, y, w, h, decayed_conf))

        # --- 法则 2：未匹配的新标签 (严苛准入) ---
        for j in new_idx:
            if not used_new[j]:
                lb_cls_id, x, y, w, h, new_conf = new_labels[j]

                # 新鲜血液考核准入
                if new_conf > tau_dynamic:
                    merged.append(new_labels[j])

    return merged


def merge_label_dirs(old_dir: str, new_dir: str, out_dir: str, round_idx: int, iou_thr: float = 0.5,
                     default_conf_old: float = 1.0) -> None:
    os.makedirs(out_dir, exist_ok=True)
    class_thresholds = calculate_global_class_thresholds(new_dir)

    old_files = glob(os.path.join(old_dir, "*.txt"))
    new_files = glob(os.path.join(new_dir, "*.txt"))
    stems = {os.path.splitext(os.path.basename(p))[0] for p in old_files + new_files}

    for stem in sorted(stems):
        old_path = os.path.join(old_dir, stem + ".txt")
        new_path = os.path.join(new_dir, stem + ".txt")
        out_path = os.path.join(out_dir, stem + ".txt")

        old_labels = _load_labels(old_path, default_conf=default_conf_old)
        new_labels = _load_labels(new_path, default_conf=1.0)

        merged = merge_image_labels(old_labels, new_labels, round_idx=round_idx,
                                    class_thresholds=class_thresholds,
                                    iou_thr=iou_thr)
        _save_labels(out_path, merged)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Adaptive Pseudo-label Evolution Fusion Strategy")
    parser.add_argument("--old", required=True, help="Directory of historical pseudo labels")
    parser.add_argument("--new", required=True, help="Directory of new pseudo labels from current model")
    parser.add_argument("--out", required=True, help="Output directory for merged labels")
    parser.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for matching")
    parser.add_argument("--round-idx", type=int, default=1, required=True,
                        help="Current training round index (1-based)")
    parser.add_argument("--default-conf-old", type=float, default=0.95,
                        help="Placeholder confidence for old labels without conf")
    args = parser.parse_args()

    merge_label_dirs(
        args.old,
        args.new,
        args.out,
        round_idx=args.round_idx,
        iou_thr=args.iou_thr,
        default_conf_old=args.default_conf_old
    )


if __name__ == "__main__":
    main()