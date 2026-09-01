#!/usr/bin/env python3
import hashlib
import json
import os
import pathlib
import socket
import subprocess
import sys


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    root = pathlib.Path(os.environ["UFO_ROOT"])
    sys.path.insert(0, str(root))
    required = [
        pathlib.Path(os.environ["UFO_VGG16_WEIGHTS"]),
        pathlib.Path(os.environ["UFO_LPIPS_CKPT"]),
        root / "offline_assets/gsplat-1.5.3-source.tar.gz",
        root / "offline_assets/data_contract/waymo_train.txt",
        root / "offline_assets/data_contract/waymo_val.txt",
        root / "offline_assets/data_contract/waymo_instance_scene_manifest.json",
        pathlib.Path(os.environ["UFO_DYNAMIC_POOL"]),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Offline assets missing: {missing}")
    train_count = len((root / "offline_assets/data_contract/waymo_train.txt").read_text().splitlines())
    val_count = len((root / "offline_assets/data_contract/waymo_val.txt").read_text().splitlines())
    if (train_count, val_count) != (798, 202):
        raise RuntimeError(f"Unexpected split sizes: train={train_count}, val={val_count}")
    for manifest_name in ("weights.sha256", "data_contract.sha256", "gsplat_portable.sha256"):
        subprocess.run(
            ["sha256sum", "--check", str(root / "offline_assets/manifests" / manifest_name)],
            cwd=root,
            check=True,
            stdout=subprocess.DEVNULL,
        )

    def deny_network(*_args, **_kwargs):
        raise RuntimeError("Network access attempted during offline readiness test")

    socket.create_connection = deny_network
    socket.socket.connect = deny_network
    from ufo.utils.lpips import LPIPS
    model = LPIPS(use_dropout=False).eval()
    del model
    print(json.dumps({
        "train_scenes": train_count,
        "validation_scenes": val_count,
        "weights": {str(path.relative_to(root)): sha256(path) for path in required[:3]},
        "asset_hashes": "PASS",
        "OFFLINE_READINESS": "PASS",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
