#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import pathlib
import shutil

import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_copy(source, destination):
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise RuntimeError(f"Frozen checkpoint already exists with different content: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary, follow_symlinks=True)
    os.chmod(temporary, 0o444)
    os.replace(temporary, destination)


def checkpoint_step(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload.get("latest_step", -1)) + 1, payload.get("best_validation_psnr")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--expected-steps", required=True, type=int)
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--stage", required=True, choices=["2s", "8s"])
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir).resolve()
    checkpoint_dir = run_dir / "checkpoints"
    best_source = checkpoint_dir / "best.pth"
    last_source = checkpoint_dir / "latest.pth"
    if not best_source.is_file() or not last_source.exists():
        raise FileNotFoundError(f"Missing best/last checkpoint in {checkpoint_dir}")
    last_step, _ = checkpoint_step(last_source)
    if last_step < args.expected_steps:
        raise RuntimeError(f"Refusing to freeze incomplete {args.stage} run: {last_step} < {args.expected_steps}")

    artifact_dir = pathlib.Path(args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    best_destination = artifact_dir / "best.pth"
    last_destination = artifact_dir / "last.pth"
    freeze_copy(best_source.resolve(), best_destination)
    freeze_copy(last_source.resolve(), last_destination)

    best_step, best_psnr = checkpoint_step(best_destination)
    parent = pathlib.Path(args.parent_checkpoint).resolve() if args.parent_checkpoint else None
    manifest = {
        "stage": args.stage,
        "source_run_dir": str(run_dir),
        "expected_optimizer_steps": args.expected_steps,
        "best": {"path": str(best_destination), "step": best_step, "validation_psnr": best_psnr, "sha256": sha256(best_destination)},
        "last": {"path": str(last_destination), "step": last_step, "sha256": sha256(last_destination)},
        "parent": None if parent is None else {"path": str(parent), "sha256": sha256(parent)},
        "immutable_mode": "0444"
    }
    manifest_path = artifact_dir / "lineage.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.chmod(manifest_path, 0o444)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
