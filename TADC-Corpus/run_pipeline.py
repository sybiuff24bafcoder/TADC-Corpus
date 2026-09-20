# -*- coding: utf-8 -*-
"""Unified TADC-Corpus pipeline entry point."""

from __future__ import annotations

import argparse

from stage1_dinov3.run_stage1 import run_stage1
from stage2_qwen.run_stage2 import run_stage2
from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="TADC-Corpus unified pipeline")
    parser.add_argument("--config", required=True, help="Path to configs/pipeline.yaml")
    parser.add_argument(
        "--stage",
        choices=["1", "2", "3", "1-2", "all"],
        default="1-2",
        help="Run one stage, Stage 1-2, or the full pipeline",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    if args.stage in {"1", "1-2", "all"}:
        run_stage1(config)
    if args.stage in {"2", "1-2", "all"}:
        run_stage2(config)
    if args.stage in {"3", "all"}:
        from stage3_yolo.run_stage3 import run_stage3
        run_stage3(config)

    print("\n" + "=" * 72)
    print("TADC-Corpus pipeline finished.")
    print("=" * 72)


if __name__ == "__main__":
    main()
