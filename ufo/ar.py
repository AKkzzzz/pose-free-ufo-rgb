# Copyright (C) 2026 Xiaomi Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import ufo.utils.misc as misc
from ufo.visualization.video_maker import make_video
import logging
import numpy as np
import torch
import os
logger = logging.getLogger("UFO")

import torch.nn.functional as F
def val(args, model, dataset, log_writer=None, output_prefix=None):

    # misc.load_model(args, model)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{args.model} Trainable Parameters: {num_trainable_params / 1e6:.2f}M")
    model.eval().cuda()
    logger.info(f"Preparing data... (This may take a while)")
    # data_dict_list = dataset.__getitem__(index=0)
    # data_dict_list = to_batch_tensor(data_dict_list)
    logger.info(f"Done preparing data.")
    # for i in range(22, 35):

    metrics_all = {
        'psnr': [],
        'ssim': [],
        'depth_rmse': [],
        'occupied_psnr': [],
        'occupied_ssim': [],
        'dynamic_psnr': [],
        'dynamic_ssim': [],
        'dynamic_depth_rmse': []
    
    }

    ### static scenes
    for i, start_idx in [(160, 0), (77, 0), (36, 0), (171, 0), (187, 0), (124, 0), (117, 0), (104, 0), (178, 0), (90, 0), (114, 0), (197, 0)]:
    ### dynamic scenes use first 20 for visualization (TODO: need to select)178，90，114，197
        if output_prefix is None:
            output_prefix = ''
        output_name = f"{output_prefix}_{i:05d}_{start_idx:03d}.mp4"
        output_dir = os.path.dirname(output_name)
        if output_dir != '' and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        val_metrics = make_video(
            args,
            dataset=dataset,
            model=model,
            device=torch.device(args.device),
            output_filename=output_name,
            scene_id=i,
            eval_metrics=True,
            start_idx=start_idx
            # data_dict=data_dict,
        )
        for key in val_metrics:
            if key not in metrics_all:
                metrics_all[key] = []
        for key in metrics_all:
            metrics_all[key].append(val_metrics[key])
        
        print(f"Saved video to {output_name}")
    

    ### TODO, summarize and export to csv

    logger.info(f"[val] Scene metrics all: {metrics_all}")

    for key in metrics_all:
        metrics_all[key] = np.mean(metrics_all[key])
    logger.info(f"[val] Scene metrics: {metrics_all}")
    if log_writer is not None:
        log_writer.update({f"val_ar/{k}": v for k, v in metrics_all.items()})
    return metrics_all
