#!/usr/bin/env python3
"""Reproducible approximation of the VGGT-Omega Sintel paper protocol.

The paper states 10 random frames per sequence, AbsRel/delta1.25 depth, and
pairwise pose AUC. Its exact sample seed and evaluation code are not public.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'benchmarks/sintel/sdk/python'))
from sintel_io import depth_read,cam_read
from demo_gradio import load_model
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

def auc(errors,threshold):
    # VGGT/DA3 protocol: integer-degree histogram followed by cumulative recall.
    histogram,_=np.histogram(np.asarray(errors),bins=np.arange(threshold+1))
    return float(100*np.mean(np.cumsum(histogram.astype(float)/len(errors))))
def angle(R): return np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))
def pose_errors(pred,gt):
    re=[];te=[]
    for i in range(len(pred)):
      for j in range(i+1,len(pred)):
        # Inputs are world-to-camera; T_j @ inv(T_i) maps camera i to camera j.
        Rp,tp=pred[i,:3,:3],pred[i,:3,3]; Rg,tg=gt[i,:3,:3],gt[i,:3,3]
        Rpr=pred[j,:3,:3]@Rp.T; tpr=pred[j,:3,3]-Rpr@tp; Rgr=gt[j,:3,:3]@Rg.T; tgr=gt[j,:3,3]-Rgr@tg
        re.append(angle(Rpr@Rgr.T)); den=np.linalg.norm(tpr)*np.linalg.norm(tgr); te.append(90 if den<1e-9 else np.degrees(np.arccos(np.clip(abs(tpr@tgr)/den,-1,1))))
    return np.maximum(re,te)
def transform_gt(depth,h,w):
    # Mirror official center crop for aspect ratio < 0.5, then resize to model output.
    H,W=depth.shape; cw=min(W,max(1,round(H/.5))); left=max((W-cw)//2,0); return cv2.resize(depth[:,left:left+cw],(w,h),interpolation=cv2.INTER_NEAREST)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--max_sequences',type=int);a=p.parse_args()
 model=load_model(str(a.checkpoint)); seqs=sorted(x.name for x in (a.root/'training/final').iterdir() if x.is_dir()); seqs=seqs[:a.max_sequences] if a.max_sequences else seqs; results=[]
 for si,seq in enumerate(seqs):
  paths=sorted((a.root/'training/final'/seq).glob('*.png')); rng=np.random.default_rng(a.seed+si); ids=np.sort(rng.choice(len(paths),min(10,len(paths)),replace=False)); chosen=[paths[i] for i in ids]
  images=load_and_preprocess_images([str(x) for x in chosen],image_resolution=512).cuda(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
  with torch.inference_mode(): pred=model(images)
  ext,_=encoding_to_camera(pred['pose_enc'],pred['images'].shape[-2:]); depth=pred['depth'][0,...,0].cpu().numpy(); ext=ext[0].cpu().numpy(); elapsed=time.perf_counter()-start
  gt_d=[];gt_e=[]
  for k,i in enumerate(ids):
   stem=paths[i].stem; gd=depth_read(str(a.root/'training/depth'/seq/f'{stem}.dpt')); gt_d.append(transform_gt(gd,*depth[k].shape)); _,ge=cam_read(str(a.root/'training/camdata_left'/seq/f'{stem}.cam')); gt_e.append(np.vstack([ge,[0,0,0,1]]))
  gt_d=np.asarray(gt_d); gt_e=np.asarray(gt_e); valid=np.isfinite(gt_d)&(gt_d>0)&np.isfinite(depth)&(depth>0); scale=float(np.median(gt_d[valid]/depth[valid])); pd=depth[valid]*scale; gd=gt_d[valid]
  pred_e=np.tile(np.eye(4),(len(ext),1,1)); pred_e[:,:3,:4]=ext; pe=pose_errors(pred_e,gt_e)
  results.append({'sequence':seq,'frame_indices':ids.tolist(),'scale':scale,'abs_rel':float(np.mean(abs(pd-gd)/gd)),'delta_1.25_percent':float(100*np.mean(np.maximum(pd/gd,gd/pd)<1.25)),'auc_3_deg':auc(pe,3),'auc_30_deg':auc(pe,30),'num_depth_pixels':int(len(gd)),'inference_seconds':elapsed,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30})
  print(json.dumps(results[-1]))
 def w(k): return float(np.mean([x[k] for x in results]))
 out={'status':'approximate reproduction: exact paper seed/splits/evaluation code are not public','dataset':'Sintel','seed':a.seed,'num_sequences':len(results),'summary':{k:w(k) for k in ['auc_3_deg','auc_30_deg','delta_1.25_percent','abs_rel']},'paper_1b':{'auc_3_deg':35.3,'auc_30_deg':73.0,'delta_1.25_percent':89.5,'abs_rel':.097},'sequences':results};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['summary'],indent=2))
if __name__=='__main__':main()
