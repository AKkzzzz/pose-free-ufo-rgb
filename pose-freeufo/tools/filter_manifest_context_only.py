#!/usr/bin/env python3
import argparse, json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--input", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()

x = json.loads(a.input.read_text())
imgs = [e for e in x["images"] if e["role"] == "context"]

x["images"] = imgs
x["pose_contract"]["name"] = "rgb_context_only_camera_prediction_v1"
x["pose_contract"]["sensor_inputs"] = ["context_rgb"]
x["pose_contract"]["target_rgb_used_for_camera"] = False

a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(x, indent=2) + "\n")

print("context images:", len(imgs))
print("frames:", sorted(set(e["frame_id"] for e in imgs)))
print("output:", a.output)
