import os
import sys
sys.path.insert(0,os.getcwd())
import numpy as np
from utils.flowlib import read_flow, flow_to_image
from utils.util_flow import write_flow, readPFM
import cv2

import scipy.optimize
import pdb
import PIL.Image as Image
from joblib import Parallel, delayed

from typing import Any, Dict, List, Tuple, Union

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', default='/data/ptmodel/',
                    help='data path')
parser.add_argument('--dataset', default='2015',
                    help='dataset name')
parser.add_argument('--method', default='ours',
                    help='{ours, detectron2, ...}')
parser.add_argument('--impath', default='',
                    help='reference image path')
parser.add_argument('--sintelpath', default='/data/gengshay/sintel_rigidity/training/rigidity/',
                    help='sintel dataset path')
parser.add_argument('--verbose', dest='verbose', action='store_true')
args = parser.parse_args()

## KITTI
# dataloader
if args.dataset == '2015':
    from dataloader import kitti15list as DA
    datapath = '/ssd/kitti_scene/training/'
elif args.dataset == '2015val':
    from dataloader import kitti15list_val as DA
    datapath = '/ssd/kitti_scene/training/'
elif args.dataset == 'sintel':
    from dataloader import sintel_mrflow_val as DA
    datapath = '/ssd/rob_flow/training/'
elif args.dataset == 'sintelval':
    from dataloader import sintellist_val as DA
    datapath = '/ssd/rob_flow/training/'

test_left_img, test_right_img ,flow_paths= DA.dataloader(datapath)
if '2015' in args.dataset:
    maskp = [i.replace('flow_occ', 'obj_map') for i in flow_paths]
else: # load Sintel data
    maskp = []
    for fp in flow_paths:
        passname = fp.split('/')[-1].split('_')[-4]
        seqname1 = fp.split('/')[-1].split('_')[-3]
        seqname2 = fp.split('/')[-1].split('_')[-2]
        framename = int(fp.split('/')[-1].split('_')[-1].split('.')[0])
        maskp.append('%s/%s_%s/frame_%04d.png'%(args.sintelpath, seqname1,seqname2,framename+1))

def eval_f(fp):
    import warnings
    warnings.filterwarnings("ignore")
    # gt
    if '2015' in args.dataset:
        mask_gt = cv2.imread(maskp[fp],0)
        flow_gt = read_flow(maskp[fp].replace('obj_map', 'flow_occ')).astype(np.float32)
        validmask = flow_gt[:,:,-1]==1
        bgmask_gt = np.logical_and(validmask, mask_gt==0).astype(float)
    else:
        bgmask_gt = cv2.imread(maskp[fp],0)>0
        shape = bgmask_gt.shape[:2]
        validmask = np.ones(shape).astype(bool)
    shape = bgmask_gt.shape[:2]

    # pred
    flowpath = flow_paths[fp]
    # KITTI
    if '2015' in args.dataset:
        idnum = int(flowpath.split('/')[-1].split('_')[-2])
        if args.method=='ours':
            maskp_pred = '%s/%s/pm-%06d_10.pfm'%(args.path, args.dataset, idnum)
            mask_pred = readPFM(maskp_pred)[0].astype(int)
            bgmask_pred = mask_pred == 0
        elif args.method=='detectron2':
            maskp_pred = '/data/gengshay/code/moseg/moseg-detectron2/%06d.png'%idnum
            mask_pred = cv2.imread(maskp_pred,0)
            bgmask_pred = mask_pred == 0
        elif args.method == 'mrflowss':
            bgmask_pred = np.load('/home/gengshay/code/SF/mrflow/semantic_rigidity_cvpr2017/output_kitti_smodel/%06d_10.npy'%idnum); 
            bgmask_pred = cv2.resize(bgmask_pred.astype(float),shape[::-1]).astype(bool)
        elif args.method == 'mrflowsk':
            bgmask_pred = cv2.imread('/home/gengshay/code/SF/mrflow/semantic_rigidity_cvpr2017/output_kitti_kmodel/%06d_10.png'%idnum,0).astype(bool)
        elif args.method == 'u2net':
            bgmask_pred = cv2.imread('/data/gengshay/code/U-2-Net/test_data/u2net_results/%06d_10.png'%idnum,0)<128
        elif args.method == 'rtn':
            bgmask_pred = ~cv2.imread('/data/gengshay/code/learningrigidity_mod/output_kitti/%06d_10.png'%idnum,0).astype(bool)
        elif args.method == 'fusionseg':
            bgmask_pred = ~np.load('/data/gengshay/code/fusionseg/python/kitti/results_liu/joint_davis_val/%06d_10_mask.npy'%idnum).astype(bool)
        elif args.method == 'fusionsegm':
            bgmask_pred = ~np.load('/data/gengshay/code/fusionseg/python/kitti/results_liu/motion/%06d_10_mask.npy'%idnum).astype(bool)
        elif args.method == 'cosnet':
            bgmask_pred = cv2.resize(cv2.imread('/data/gengshay/code/COSNet/result/test/davis_iteration_conf/Results/kitti/%06d_10.png'%idnum,0), shape[::-1])<128
        elif args.method == 'matnet':
            bgmask_pred = cv2.imread('/data/gengshay/code/MATNet/output/kitti_liu/%06d_10.png'%idnum,0)<128
        elif args.method == 'cc':
            bgmask_pred = cv2.resize(np.load('/data/gengshay/code/cc/output/mask-joint/%03d.npy'%idnum)[0],shape[::-1])==1
        elif args.method == 'ccm':
            bgmask_pred = cv2.resize(np.load('/data/gengshay/code/cc/output/mask-motion/%03d.npy'%idnum)[0],shape[::-1])==1
        elif args.method == 'angle':
            maskp_pred = '%s/%s/pm-%06d_10.pfm'%(args.path, args.dataset, idnum)
            mask_pred = readPFM(maskp_pred)[0].astype(int)
            bgmask_pred = mask_pred == 0
        else:exit()
    
    # Sintel
    else:
        passname = flowpath.split('/')[-1].split('_')[-4]
        seqname1 = flowpath.split('/')[-1].split('_')[-3]
        seqname2 = flowpath.split('/')[-1].split('_')[-2]
        framename = int(flowpath.split('/')[-1].split('_')[-1].split('.')[0])
        if args.method=='ours':
            maskp_pred = '%s/%s/pm-Sintel_%s_%s_%s_%02d.pfm'%(args.path, args.dataset, passname, seqname1,seqname2,framename)
            mask_pred = readPFM(maskp_pred)[0]  # label map in {1,...N} given N rigid body predictions
            bgmask_pred = mask_pred == 0
        elif args.method=='detectron2':
            maskp_pred = '/data/gengshay/code/moseg/smoseg-detectron2/Sintel_%s_%s_%s_%02d.png'%(passname, seqname1,seqname2,framename)
            mask_pred = cv2.imread(maskp_pred,0)
            bgmask_pred = mask_pred == 0
        elif args.method == 'mrflowss':
            bgmask_pred = np.load('/home/gengshay/code/SF/mrflow/semantic_rigidity_cvpr2017/output_sintel_smodel/Sintel_%s_%s_%s_%02d.npy'%(passname,seqname1,seqname2,framename)).astype(bool)
        elif args.method == 'mrflowsk':
            bgmask_pred = cv2.imread('/home/gengshay/code/SF/mrflow/semantic_rigidity_cvpr2017/output_sintel_kmodel/Sintel_%s_%s_%s_%02d.png'%(passname,seqname1,seqname2,framename),0).astype(bool)
        elif args.method == 'u2net':
            bgmask_pred = cv2.imread('/data/gengshay/code/U-2-Net/test_data/u2net_results_sintel/Sintel_%s_%s_%s_%02d.png'%(passname,seqname1,seqname2,framename),0)<128
        elif args.method == 'rtn':
            bgmask_pred = ~cv2.imread('/data/gengshay/code/learningrigidity_mod/output_sintel/Sintel_%s_%s_%s_%02d.png'%(passname,seqname1,seqname2,framename),0).astype(bool)
        elif args.method == 'fusionseg':
            bgmask_pred = ~np.load('/data/gengshay/code/fusionseg/python/sintel/results/joint_davis_val/Sintel_%s_%s_%s_%02d_mask.npy'%(passname,seqname1,seqname2,framename)).astype(bool)
        elif args.method == 'fusionsegm':
            bgmask_pred = ~np.load('/data/gengshay/code/fusionseg/python/sintel/results/motion/Sintel_%s_%s_%s_%02d_mask.npy'%(passname,seqname1,seqname2,framename)).astype(bool)
        elif args.method == 'cosnet':
            bgmask_pred = cv2.resize(cv2.imread('/data/gengshay/code/COSNet/result/test/davis_iteration_conf/Results/sintel/Sintel_%s_%s_%s_%02d.png'%(passname,seqname1,seqname2,framename),0), shape[::-1])<128
        elif args.method == 'matnet':
            bgmask_pred = cv2.imread('/data/gengshay/code/MATNet/output/sintel/Sintel_%s_%s_%s_%02d.png'%(passname,seqname1,seqname2,framename),0)<128
        elif args.method == 'angle':
            maskp_pred = '%s/%s/pm-Sintel_%s_%s_%s_%02d.pfm'%(args.path, args.dataset, passname, seqname1,seqname2,framename)
            mask_pred = readPFM(maskp_pred)[0]
            bgmask_pred = mask_pred == 0
        else:exit() 
        
    if args.method!='ours' and args.method!='detectron2':    
        _, mask_pred =cv2.connectedComponentsWithAlgorithm((1-bgmask_pred.astype(np.uint8)),connectivity=8,ltype=cv2.CV_16U,ccltype=cv2.CCL_WU)

    # bg-iou
    try:
        bgiou = np.logical_and(bgmask_gt.astype(bool)[validmask],bgmask_pred[validmask]).sum() / np.logical_or(bgmask_gt.astype(bool)[validmask],bgmask_pred[validmask]).sum()
    except:pdb.set_trace() 
   
    # obj-F (penalizes false positive)
    # defined in https://arxiv.org/abs/1902.03715 Sec.4
    # F=2PR/P+R => 2/(1/P+1/R)
    # since P = sum(|c and g|) / sum(|c|) and R = sum(|c and g|) / sum(|g|), 
    #    where  c is prediction and g is GT
    # 1/P+1/R = [sum(|c|) + sum(|g|)] / sum(|c and g|) 
    # therefore F = 2*sum(|c and g|) / [sum(|c|) +  sum(|g|)]
    if '2015' in args.dataset:
        gtlist = list(set(mask_gt.flatten()))
        M = len(gtlist)-1
        imatrix = np.zeros((M,mask_pred.max()))
        fmatrix = np.zeros((M,mask_pred.max()))
        for i in range(M):  # for all bg instances
            objx_mask_gt=mask_gt==gtlist[i+1]
            for j in range(mask_pred.max()):
                objx_mask_pred = mask_pred == j+1
                imatrix[i,j] = float(np.logical_and(objx_mask_gt,objx_mask_pred)[validmask].sum())
                fmatrix[i,j] = imatrix[i,j]/(objx_mask_gt[validmask].sum()+objx_mask_pred[validmask].sum())*2
        ridx,cidx = scipy.optimize.linear_sum_assignment(1-fmatrix)
        objf = imatrix[ridx, cidx].sum() / ((mask_pred>0)[validmask].sum()+(mask_gt>0)[validmask].sum()) *2
    else:
        objf = 0.

    return bgiou, objf

rt = Parallel(n_jobs=1)(delayed(eval_f)(fp) for fp in range(len(test_left_img)) )
ious = [k[0] for k in rt]
objf = [k[1] for k in rt]

if args.verbose:
    for i in np.argsort([-i for i in ious]):
        print('%.2f/%.2f/%s'%(ious[i],objf[i], maskp[i]))
print('%d images'%len(ious))
print('IoU: %.2f'%(np.mean(ious)*100))
print('obj: %.2f'%(np.mean(objf)*100))
