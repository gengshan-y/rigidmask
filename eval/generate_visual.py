import os
import sys
sys.path.insert(0,os.getcwd())
import cv2
import torch
import glob
import numpy as np
import pdb
import imageio

import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from typing import Any, Dict, List, Tuple, Union
from utils.util_flow import readPFM

coco_metadata = MetadataCatalog.get("coco_2017_val")

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--datapath', default='',
                    help='dataset path')
parser.add_argument('--imgpath', default='',
                    help='dataset path')
args = parser.parse_args()

class Object(object):
    def has(self, name: str) -> bool:
        return name in self._fields
    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

frames=[]
for i,path in enumerate(sorted(glob.glob('%s/pm*'%args.datapath))):
    print(path)
    pred =  readPFM(path)[0]
    center_img = cv2.imread(path.replace('pm', 'mvis').replace('.pfm', '.jpg'))
    img = cv2.imread('%s/%s.png'%(args.imgpath,path.split('/')[-1].split('pm-')[1].split('.pfm')[0]))
    if img is None:
        img = cv2.imread('%s/%s.jpg'%(args.imgpath,path.split('/')[-1].split('pm-')[1].split('.pfm')[0]))
    shape = pred.shape[:2]
    num_instances = int(pred.max())

    # if no object detected
    if num_instances==0:
        _, pred =cv2.connectedComponentsWithAlgorithm((1-(pred==0).astype(np.uint8)),connectivity=8,ltype=cv2.CV_16U,ccltype=cv2.CCL_WU)
        num_instances = pred.max()

    if num_instances>0:
        pred_masks = torch.zeros((num_instances,)+shape).bool()
        for k in range(num_instances):
            pred_masks[k] = torch.Tensor(pred==(k+1))

        obj = Object()
        obj.image_height = shape[0]
        obj.image_width =  shape[1]
        obj._fields = {}
        obj._fields["pred_masks"] = pred_masks

        v = Visualizer(img, coco_metadata, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
        try:
            vis = v.draw_instance_predictions(obj)
        except:pdb.set_trace()
        mask_result = vis.get_image()
    else:
        mask_result = cv2.resize(img,None,fx=0.5,fy=0.5)
        
    # write results
    cv2.imwrite(path.replace('pm-', 'vis-').replace('.pfm','.png'),  mask_result)
    try:
        center_img = cv2.resize(center_img, mask_result.shape[:2][::-1])
        blend = cv2.addWeighted(mask_result, 1, center_img, 1, 0)
        cv2.imwrite(path.replace('pm-', 'bvis-').replace('.pfm','.png'), blend)
    except:pass
    frame = blend[:,:,::-1].copy()
    frames.append(frame)

imageio.mimsave('./output-seg.gif', frames, duration=5./len(frames))
