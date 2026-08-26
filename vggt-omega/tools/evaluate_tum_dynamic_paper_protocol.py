#!/usr/bin/env python3
"""Reproducible 10-frame VGGT-Omega evaluation on TUM-Dynamic."""
import argparse,json,sys,time
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from demo_gradio import load_model
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera
from evaluate_sintel_paper_protocol import auc,pose_errors
def read(p):
 out={}
 for l in p.read_text().splitlines():
  if l and not l.startswith('#'):
   x=l.split();out[float(x[0])]=x[1:]
 return out
def nearest(t,d,maxdiff=.02):
 k=min(d,key=lambda x:abs(x-t));return k if abs(k-t)<maxdiff else None
def resize_depth(gt,h,w): return cv2.resize(gt,(w,h),interpolation=cv2.INTER_NEAREST)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=0);a=p.parse_args();model=load_model(str(a.checkpoint));res=[]
 for si,s in enumerate(sorted(a.root.glob('rgbd_dataset_freiburg3_*'))):
  if not s.is_dir():continue
  rgb,dep,pose=read(s/'rgb.txt'),read(s/'depth.txt'),read(s/'groundtruth.txt'); candidates=[]
  for t in sorted(rgb):
   td,tp=nearest(t,dep),nearest(t,pose)
   if td is not None and tp is not None:candidates.append((t,td,tp))
  rng=np.random.default_rng(a.seed+si); ids=np.sort(rng.choice(len(candidates),10,replace=False)); sel=[candidates[i] for i in ids]; paths=[s/rgb[t][0] for t,_,_ in sel]
  im=load_and_preprocess_images([str(x) for x in paths],image_resolution=512).cuda();torch.cuda.reset_peak_memory_stats();st=time.perf_counter()
  with torch.inference_mode():o=model(im)
  ext,_=encoding_to_camera(o['pose_enc'],o['images'].shape[-2:]);ext=ext[0].cpu().numpy();pd=o['depth'][0,...,0].cpu().numpy();gt_d=[];gt_e=[]
  for k,(tr,td,tp) in enumerate(sel):
   gd=cv2.imread(str(s/dep[td][0]),cv2.IMREAD_UNCHANGED).astype(np.float32)/5000.;gt_d.append(resize_depth(gd,*pd[k].shape)); q=np.asarray(pose[tp],float);x,y,z,qx,qy,qz,qw=q
   from scipy.spatial.transform import Rotation
   M=np.eye(4);M[:3,:3]=Rotation.from_quat([qx,qy,qz,qw]).as_matrix();M[:3,3]=[x,y,z];gt_e.append(np.linalg.inv(M))
  gt_d=np.asarray(gt_d);gt_e=np.asarray(gt_e);v=(gt_d>0)&np.isfinite(gt_d)&(pd>0)&np.isfinite(pd);scale=float(np.median(gt_d[v]/pd[v]));x=pd[v]*scale;y=gt_d[v];P=np.tile(np.eye(4),(10,1,1));P[:,:3,:4]=ext;e=pose_errors(P,gt_e)
  res.append({'sequence':s.name,'rgb_timestamps':[x[0] for x in sel],'abs_rel':float(np.mean(abs(x-y)/y)),'delta_1.25_percent':float(100*np.mean(np.maximum(x/y,y/x)<1.25)),'auc_3_deg':auc(e,3),'auc_30_deg':auc(e,30),'inference_seconds':time.perf_counter()-st,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30});print(json.dumps(res[-1]))
 summary={k:float(np.mean([x[k] for x in res])) for k in ['auc_3_deg','auc_30_deg','delta_1.25_percent','abs_rel']};out={'status':'approximate reproduction: exact paper seed/evaluation code are not public','dataset':'TUM-Dynamic','seed':a.seed,'summary':summary,'paper_1b':{'auc_3_deg':30.2,'auc_30_deg':82.3,'delta_1.25_percent':97.4,'abs_rel':.041},'sequences':res};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(summary)
if __name__=='__main__':main()
