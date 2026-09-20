# -*- coding: utf-8 -*-
"""TADC-Corpus Stage 3: YOLO iterative pseudo-label refinement."""

from __future__ import annotations

import argparse
import os
import shutil
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ultralytics import YOLO

from stage3_yolo.pseudo_label_merge import (
    labels5_to_labels6,
    labels6_to_labels5,
    merge_label_dirs,
)
from utils.config import load_config


def get_round_subdir(work_dir: Path, round_idx: int, sub: str) -> Path:
    d = work_dir / f"round{round_idx}" / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def setup_dataset(config: Dict[str, Any], work_dir: Path) -> Tuple[Path, Path, Path]:
    cfg = config["stage3"]
    root = Path(cfg["dataset_root"])
    train_images = root / "images" / "train"
    val_images = root / "images" / "val"
    if not train_images.is_dir():
        raise FileNotFoundError(f"Training image directory not found: {train_images}")
    if not val_images.is_dir():
        raise FileNotFoundError(f"Validation image directory not found: {val_images}")

    data_yaml = work_dir / "data_unsup.yaml"
    with data_yaml.open("w", encoding="utf-8") as f:
        f.write(f"path: {str(root).replace('\\', '/') }\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for idx, name in enumerate(config["dataset"]["classes"]):
            f.write(f"  {idx}: {name}\n")
    return root, data_yaml, train_images


def copy_train_labels_to_dataset(dataset_root: Path, src_dir: Path) -> None:
    dst = dataset_root / "labels" / "train"
    dst.mkdir(parents=True, exist_ok=True)
    for f in dst.glob("*.txt"):
        f.unlink()
    for txt in src_dir.glob("*.txt"):
        shutil.copy2(txt, dst / txt.name)


def train_one_round(cfg: Dict[str, Any], work_dir: Path, data_yaml: Path, round_idx: int, weights_path: str) -> str:
    schedule = cfg.get("round_schedule", {})
    item = schedule.get(str(round_idx)) or schedule.get(round_idx)
    if item is None:
        item = {"epochs": 30, "lr0": 0.0002, "close_mosaic": 5, "warmup_epochs": 2.0}

    model = YOLO(weights_path)
    run_name = f"unsup_round{round_idx}"
    model.train(
        data=str(data_yaml),
        epochs=int(item["epochs"]),
        imgsz=int(cfg["image_size"]),
        batch=int(cfg["batch_size"]),
        project=str(work_dir),
        workers=int(cfg["workers"]),
        name=run_name,
        exist_ok=True,
        optimizer="SGD",
        lr0=float(item["lr0"]),
        close_mosaic=int(item["close_mosaic"]),
        warmup_epochs=float(item["warmup_epochs"]),
        mosaic=1.0,
        mixup=0.1,
        weight_decay=0.0005,
    )
    best = work_dir / run_name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"best.pt not found at {best}")
    return str(best)


def generate_predictions(weights_path: str, train_images_dir: Path, out_dir: Path, cfg: Dict[str, Any]) -> None:
    model = YOLO(weights_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_size = int(cfg["image_size"])
    conf_thres = float(cfg["prediction_conf"])
    for result in model(
        str(train_images_dir),
        imgsz=image_size,
        conf=conf_thres,
        stream=True,
        verbose=False,
        augment=True,
    ):
        boxes = result.boxes
        stem = Path(result.path).stem
        out_txt = out_dir / f"{stem}.txt"
        with out_txt.open("w", encoding="utf-8") as f_out:
            if boxes is None or len(boxes) == 0:
                continue
            xywhn = boxes.xywhn.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            for cls_id, box, conf in zip(cls_ids, xywhn, confs):
                f_out.write(
                    f"{int(cls_id)} {float(box[0]):.6f} "
                    f"{float(box[1]):.6f} {float(box[2]):.6f} "
                    f"{float(box[3]):.6f} {float(conf):.6f}\n"
                )


def run_stage3(config: Dict[str, Any]) -> str:
    cfg = config["stage3"]
    output_root = Path(config["paths"]["output_root"])
    work_dir = Path(cfg["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)

    dataset_root, data_yaml, train_images_dir = setup_dataset(config, work_dir)
    initial_labels_dir = Path(
        config["paths"].get(
            "stage2_yolo_labels",
            str(output_root / "stage2" / "yolo_labels"),
        )
    )
    if not initial_labels_dir.is_dir():
        raise FileNotFoundError(f"Stage-2 labels not found: {initial_labels_dir}")

    num_rounds = int(cfg["num_rounds"])
    current_weights = str(cfg["base_model"])
    history6_dir = work_dir / "history_round0_6"
    labels5_to_labels6(
        initial_labels_dir,
        history6_dir,
        default_conf=float(cfg["lmm_init_conf"]),
    )

    for round_idx in range(1, num_rounds + 1):
        print(f"\n{'=' * 72}\nRound {round_idx} / {num_rounds}\n{'=' * 72}")
        os.environ["CURRENT_ROUND"] = str(round_idx)
        train_labels5_dir = get_round_subdir(work_dir, round_idx, "train_labels5")
        labels6_to_labels5(history6_dir, train_labels5_dir)
        copy_train_labels_to_dataset(dataset_root, train_labels5_dir)

        current_weights = train_one_round(
            cfg,
            work_dir,
            data_yaml,
            round_idx,
            current_weights,
        )

        if round_idx == num_rounds:
            break

        preds6_dir = get_round_subdir(work_dir, round_idx, "pred6")
        generate_predictions(current_weights, train_images_dir, preds6_dir, cfg)
        merged6_dir = get_round_subdir(work_dir, round_idx, "merged6")
        merge_label_dirs(
            str(history6_dir),
            str(preds6_dir),
            str(merged6_dir),
            round_idx=round_idx,
            iou_thr=float(cfg["iou_merge_thr"]),
            default_conf_old=float(cfg["lmm_init_conf"]),
        )
        history6_dir = merged6_dir

    print(f"\n[Done] Stage 3 finished. Final weights: {current_weights}")
    return current_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="TADC-Corpus Stage 3")
    parser.add_argument("--config", required=True, help="Path to pipeline.yaml")
    args = parser.parse_args()
    run_stage3(load_config(args.config))


if __name__ == "__main__":
    main()
