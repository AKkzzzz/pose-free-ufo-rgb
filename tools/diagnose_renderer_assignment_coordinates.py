#!/usr/bin/env python3
"""Measure assignment GT coordinates at the renderer class-loss site."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import inference as ufo_inference
from ufo.dataset.data_utils import prepare_inputs_and_targets, to_batch_tensor
from ufo.utils.config import merge_config_and_args
from ufo.utils.diagnostics import assignment_metrics
from ufo.utils.misc import update_scene


def build_args():
    parser = ufo_inference.get_args_parser()
    parser.description = "Renderer-time assignment coordinate diagnostic"
    args = merge_config_and_args(parser, config_path=None)
    args = merge_config_and_args(parser, config_path=args.config)
    args = ufo_inference.add_missing_config_values(args, args.config)
    args.inference_assignment_mode = "predicted"
    args.renderer_assignment_coordinate_diagnostics = True
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    args = build_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = ufo_inference.build_model(args, device)
    model.args.renderer_assignment_coordinate_diagnostics = True
    dataset = ufo_inference.build_dataset(args)
    dataset_index = 0 if args.annotation_file else args.scene_id
    raw = dataset.__getitem__(dataset_index, args.start_idx, return_all=True)
    raw = to_batch_tensor(raw)
    chunks = prepare_inputs_and_targets(
        raw, device, timespan=args.timespan, from_list=True, args=args
    )

    scene = {}
    rows = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for chunk_index, (input_dict, _) in enumerate(chunks):
            input_has_local_corners = input_dict.get("context_instances_corner_local") is not None
            output, scene = update_scene(
                input_dict,
                model,
                scene=scene,
                render=True,
                filter_num=args.filter_num,
                collect_diagnostics=False,
            )
            row = {"chunk": chunk_index, "scene_context_timesteps": int(
                output["gs_params"]["means"].shape[1]
            )}
            row.update(assignment_metrics(output))
            if "renderer_global_dynamic_gt_count" not in row:
                raise RuntimeError(
                    "renderer coordinate diagnostics did not execute at class_loss; "
                    f"input_has_local_corners={input_has_local_corners}, "
                    f"output_has_local_corners="
                    f"{output.get('context_instances_corner_local') is not None}, "
                    f"model_flag={getattr(model.args, 'renderer_assignment_coordinate_diagnostics', None)}, "
                    f"renderer_keys={[key for key in output if key.startswith('renderer_')]}"
                )
            rows.append(row)

    summary = {
        "scene_id": args.scene_id,
        "start_idx": args.start_idx,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "measurement_site": "forward_renderer class_loss",
        "chunks": rows,
    }
    output_path = output_dir / (
        f"renderer_assignment_coordinates_start_{args.start_idx:03d}.json"
    )
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
