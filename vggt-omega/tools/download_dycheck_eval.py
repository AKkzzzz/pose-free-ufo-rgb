#!/usr/bin/env python3
"""Download only the seven DyCheck iPhone evaluation scenes and required fields."""
from __future__ import annotations

import argparse
import importlib
import subprocess
import time
from pathlib import Path

import gdown
import numpy as np
from gdown.download import _get_session


IPHONE_FOLDER = "1cBw3CUKu2sWQfc_1LbFZGbpdQyTFzDEX"
SCENES = {"apple", "block", "paper-windmill", "space-out", "spin", "teddy", "wheel"}
folder_api = importlib.import_module("gdown.download_folder")


def children(session, folder_id):
    if hasattr(folder_api, "_parse_embedded_folder_view"):
        _, entries = folder_api._parse_embedded_folder_view(session, folder_id)
    else:
        response = session.get(f"https://drive.google.com/drive/folders/{folder_id}?hl=en")
        response.raise_for_status()
        _, entries = folder_api._parse_google_drive_file(response.url, response.text)
    return {name: (item_id, item_type) for item_id, name, item_type in entries}


def folder_id(session, parent_id, *path):
    current = parent_id
    for part in path:
        current = children(session, current)[part][0]
    return current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", default=sorted(SCENES), choices=sorted(SCENES))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    session = _get_session(proxy=None, use_cookies=False, user_agent="Mozilla/5.0")
    if isinstance(session, tuple):
        session = session[0]
    scene_entries = children(session, IPHONE_FOLDER)
    for scene in args.scenes:
        scene_id = scene_entries[scene][0]
        targets = {
            "camera": folder_id(session, scene_id, "camera"),
            "rgb/2x": folder_id(session, scene_id, "rgb", "2x"),
            "depth/2x": folder_id(session, scene_id, "depth", "2x"),
        }
        listings = {relative: children(session, remote_id) for relative, remote_id in targets.items()}
        stems = set(Path(name).stem for name in listings["camera"])
        stems &= set(Path(name).stem for name in listings["rgb/2x"])
        stems &= set(Path(name).stem for name in listings["depth/2x"])
        stems = sorted(stems)
        rng = np.random.default_rng(args.seed + sorted(SCENES).index(scene))
        selected = sorted(rng.choice(stems, 10, replace=False).tolist())
        print(f"{scene}: selected {selected}", flush=True)
        extensions = {"camera": ".json", "rgb/2x": ".png", "depth/2x": ".npy"}
        for relative in targets:
            output = args.output / scene / relative
            output.mkdir(parents=True, exist_ok=True)
            for stem in selected:
                name = stem + extensions[relative]
                item_id = listings[relative][name][0]
                destination = output / name
                if destination.is_file():
                    continue
                result = None
                for attempt in range(10):
                    try:
                        subprocess.run(
                            [
                                "curl", "-L", "--fail", "--retry", "5", "--connect-timeout", "15",
                                "-sS", "-o", str(destination),
                                f"https://drive.usercontent.google.com/download?id={item_id}&export=download&confirm=t",
                            ],
                            check=True,
                        )
                        result = str(destination)
                    except Exception as error:
                        print(f"retry {attempt + 1}/10 for {destination.name}: {error}", flush=True)
                    if result is not None:
                        break
                    time.sleep(min(2**attempt, 30))
                if result is None:
                    raise RuntimeError(f"Failed to download {scene}/{relative}/{name}")


if __name__ == "__main__":
    main()
