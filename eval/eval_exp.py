import glob
import os
import sys
sys.path.insert(0,os.getcwd())
import numpy as np
from matplotlib import pyplot as plt
from utils.flowlib import read_flow, flow_to_image
from utils.util_flow import write_flow, readPFM
import cv2

import pdb
import PIL.Image as Image
from dataloader.robloader import disparity_loader
from utils.sintel_io import disparity_read
from joblib import Parallel, delayed

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', default='/data/ptmodel/',
                    help='database')
parser.add_argument('--dataset', default='2015',
                    help='database')
args = parser.parse_args()

## KITTI
# dataloader
if args.dataset == '2015':
    from dataloader import kitti15list as DA
    datapath = '/ssd/kitti_scene/training/'
elif args.dataset == '2015val':
    from dataloader import kitti15list_val as DA
    datapath = '/ssd/kitti_scene/training/'
elif args.dataset == '2015vallidar':
    from dataloader import kitti15list_val_lidar as DA
    datapath = '/ssd/kitti_scene/training/'
elif args.dataset == 'sintelval':
    from dataloader import sintellist_val as DA
    datapath = '/ssd/rob_flow/training/'

test_left_img, test_right_img ,flow_paths= DA.dataloader(datapath)
expansionp = [i.replace('flow_occ','expansion').replace('.png', '.pfm') for i in flow_paths]
if '2015' in args.dataset:
    disp0p = [i.replace('flow_occ','disp_occ_0') for i in flow_paths]
    disp1p = [i.replace('flow_occ','disp_occ_1') for i in flow_paths]
else:
    disp0p = []
    disp1p = []
    for fp in flow_paths:
        seqname1 = fp.split('/')[-1].split('_')[-3]
        seqname2 = fp.split('/')[-1].split('_')[-2]
        framename = int(fp.split('/')[-1].split('_')[-1].split('.')[0])
        disp0p.append('%s/disparities/%s_%s/frame_%04d.png'%(fp.rsplit('/',2)[0], seqname1, seqname2,framename+1))
        disp1p.append('%s/disparities/%s_%s/frame_%04d.png'%(fp.rsplit('/',2)[0], seqname1, seqname2,framename+2))

def eval_f(fp):
    import warnings
    warnings.filterwarnings("ignore")
    # gt
    gt_oe = disparity_loader(expansionp[fp])
    gt_logexp = -np.log(gt_oe)
    oemask = gt_oe>0

    if '2015' in args.dataset:
        gt_disp0 = disparity_loader(disp0p[fp])
        gt_disp1 = disparity_loader(disp1p[fp])
    elif args.dataset == 'sintelval':
        gt_disp0 = disparity_read(disp0p[fp])
        gt_disp1 = disparity_read(disp1p[fp])
    gt_logdc = np.log(gt_disp0/gt_disp1)
    d1mask = gt_disp0>0
    d2mask = gt_disp1>0
    dcmask = np.logical_and(d1mask,d2mask)
    dcmask = np.logical_and(dcmask, np.abs(gt_logdc)<np.log(2))

    # pred
    logexp = disparity_loader( '%s/%s/exp-%s.pfm'%(args.path,args.dataset,expansionp[fp].split('/')[-1].split('.')[0]))
    logexp = np.clip(logexp,-np.log(2),np.log(2))
    logexp_error = np.abs(gt_logexp-logexp)[oemask].mean()

    logdc = disparity_loader( '%s/%s/mid-%s.pfm'%(args.path,args.dataset,expansionp[fp].split('/')[-1].split('.')[0]))
    logdc = np.clip(logdc,-np.log(2),np.log(2))
    logmid_err = np.abs(gt_logdc-logdc)[dcmask].mean()
    return logexp_error, logmid_err
    

rt = Parallel(n_jobs=1)(delayed(eval_f)(fp) for fp in range(len(test_left_img)) )
logexp_error = [k[0] for k in rt]
logmid_error = [k[1] for k in rt]

print('logexp-err:\t%.1f (1e4)'%(10000*np.mean(logexp_error)))
print('logmid-err:\t%.1f (1e4)'%(10000*np.mean(logmid_error)))
