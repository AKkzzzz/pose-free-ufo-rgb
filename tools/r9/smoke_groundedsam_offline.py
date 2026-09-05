#!/usr/bin/env python3
"""Load the exact offline SAM2-large and GroundingDINO-tiny runtime."""

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GSAM = ROOT / "third_party/Grounded-SAM-2"
sys.path.insert(0, str(GSAM))

import torch
import supervision
import transformers
from sam2.build_sam import build_sam2_video_predictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def main():
    expected_hf = ROOT / "third_party/hf_cache"
    if Path(os.environ.get("HF_HOME", "")).resolve() != expected_hf.resolve():
        raise RuntimeError(f"HF_HOME must be {expected_hf}")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("Offline environment flags are required")
    arch = torch.cuda.get_arch_list()
    if "sm_90" not in arch:
        raise RuntimeError(f"Torch lacks sm_90: {arch}")
    checkpoint = GSAM / "checkpoints/sam2.1_hiera_large.pt"
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_l.yaml", str(checkpoint), device="cuda"
    )
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    grounding = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-tiny"
    ).to("cuda")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"torch_arch_list={arch}")
    print(f"transformers={transformers.__version__}")
    print(f"supervision={supervision.__version__}")
    print(f"sam2_predictor={type(predictor).__name__}")
    print(f"grounding_model={type(grounding).__name__} processor={type(processor).__name__}")
    print("GROUNDEDSAM_OFFLINE_LOAD=PASS")


if __name__ == "__main__":
    main()
