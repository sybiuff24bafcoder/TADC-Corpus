# -*- coding: utf-8 -*-
"""Setup helper for external repositories and requirements."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_cmd(cmd, cwd=None):
    print(f">> Executing: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd)


def setup():
    dinov3_dir = ROOT / "external" / "dinov3"
    dinov3_dir.parent.mkdir(parents=True, exist_ok=True)

    if not dinov3_dir.exists():
        print("Cloning DINOv3 repo...")
        run_cmd(["git", "clone", "https://github.com/facebookresearch/dinov3.git", str(dinov3_dir)])
    else:
        print(f"DINOv3 directory already exists at {dinov3_dir}")

    # Create weight dir
    weight_dir = dinov3_dir / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)
    print(f"Prepared DINOv3 checkpoint directory at: {weight_dir}")


if __name__ == "__main__":
    setup()