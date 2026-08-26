#!/usr/bin/env python3
"""Evaluate VGGT depth against projected sparse Waymo LiDAR depth.

Uses the VGGT-Omega paper's depth metrics (AbsRel and delta1.25), but this is
not an official paper benchmark because the paper does not evaluate Waymo.
"""
import argparse,json
from pathlib import Path
import cv2,numpy as np

def metrics(pred,gt):
    ratio=np.maximum(pred/gt,gt/pred)
    return {'abs_rel':float(np.mean(np.abs(pred-gt)/gt)),'rmse_m':float(np.sqrt(np.mean((pred-gt)**2))),'delta_1.25_percent':float(100*np.mean(ratio<1.25)),'delta_1.25_sq_percent':float(100*np.mean(ratio<1.25**2)),'delta_1.25_cu_percent':float(100*np.mean(ratio<1.25**3)),'num_lidar_pixels':int(len(gt))}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--predictions',type=Path,required=True); p.add_argument('--scene_dir',type=Path,required=True); p.add_argument('--camera',type=int,default=0); p.add_argument('--num_frames',type=int,default=10); p.add_argument('--seed',type=int,default=0); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    d=np.load(a.predictions)['depth'][...,0]; n=min(len(d),len(list((a.scene_dir/'depth_flows_4').glob(f'*_{a.camera}.npy')))); rng=np.random.default_rng(a.seed); ids=np.sort(rng.choice(n,min(a.num_frames,n),replace=False))
    ps=[]; gs=[]; per=[]
    for i in ids:
        gt=np.load(a.scene_dir/'depth_flows_4'/f'{i:03d}_{a.camera}.npy')[...,0].astype(np.float32); pr=cv2.resize(d[i].astype(np.float32),(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_LINEAR)
        m=np.isfinite(gt)&(gt>0)&np.isfinite(pr)&(pr>0); ps.append(pr[m]); gs.append(gt[m]); per.append((i,pr[m],gt[m]))
    pp=np.concatenate(ps); gg=np.concatenate(gs)
    # VGGT depth is scale ambiguous. Report one scale for the entire clip, plus
    # a per-frame oracle diagnostic that removes temporal scale inconsistency.
    global_scale=float(np.median(gg)/np.median(pp)); global_result=metrics(pp*global_scale,gg)
    oracle=[]
    for i,pr,gt in per:
        s=float(np.median(gt)/np.median(pr)); oracle.append({'frame':int(i),'scale':s,**metrics(pr*s,gt)})
    total=sum(x['num_lidar_pixels'] for x in oracle)
    weighted={k:float(sum(x[k]*x['num_lidar_pixels'] for x in oracle)/total) for k in ['abs_rel','rmse_m','delta_1.25_percent','delta_1.25_sq_percent','delta_1.25_cu_percent']}
    out={'protocol':'VGGT-Omega paper depth metrics on projected sparse Waymo LiDAR; custom Waymo diagnostic, not an official paper benchmark','sampled_frames':ids.tolist(),'camera':a.camera,'global_clip_scale':global_scale,'global_scale_metrics':global_result,'per_frame_median_scale_oracle_metrics':weighted,'per_frame':oracle}
    a.output.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
