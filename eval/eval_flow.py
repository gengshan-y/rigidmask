import os
import sys
sys.path.insert(0,os.getcwd())
from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.insert(0,'utils')
from utils.flowlib import flow_to_image, read_flow, compute_color, visualize_flow
from utils.io import mkdir_p
import pdb
import glob

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', default='/data/ptmodel/',
                    help='database')
parser.add_argument('--vis', default='no',
                    help='database')
parser.add_argument('--dataset', default='2015',
                    help='database')
args = parser.parse_args()

aepe_s = []
fall_s64 = []
fall_s32 = []
fall_s16 = []
fall_s8 = []
fall_s = []
oor_tp = []
oor_fp = []

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
elif args.dataset == '2015test':
    from dataloader import kitti15list as DA
    datapath = '/ssd/kitti_scene/testing/'
elif args.dataset == 'sintel':
    from dataloader import sintellist_val as DA
    datapath = '/ssd/rob_flow/training/'
elif args.dataset == 'sintelval':
    from dataloader import sintellist_val as DA
    datapath = '/ssd/rob_flow/training/'
elif args.dataset == 'mosegsintel':
    from dataloader import moseg_sintellist_val as DA
    datapath = '/ssd/rob_flow/training/'
elif args.dataset == 'sinteltest':
    from dataloader import sintellist as DA
    datapath = '/ssd/rob_flow/test/'
elif args.dataset == 'mb':
    from dataloader import mblist as DA
    datapath = '/ssd/rob_flow/training/'
elif args.dataset == 'hd1k':
    from dataloader import hd1klist_val as DA
    datapath = '/ssd/rob_flow/training/'
elif args.dataset == 'viper':
    from dataloader import viperlist_val as DA
    datapath = '/data/gengshay/code/rvc_devkit/flow/temp_conversion_dir/0/training/'
elif args.dataset == 'chairs':
    from dataloader import chairslist as DA
    datapath = '/ssd/FlyingChairs_release/data/'
test_left_img, test_right_img ,flow_paths= DA.dataloader(datapath)

if args.dataset == 'chairs':
    with open('misc/FlyingChairs_train_val.txt', 'r') as f:
        split = [int(i) for i in f.readlines()]
    test_left_img = [test_left_img[i]   for i,flag in enumerate(split) if flag==2]
    test_right_img = [test_right_img[i] for i,flag in enumerate(split) if flag==2]
    flow_paths = [flow_paths[i]         for i,flag in enumerate(split) if flag==2]

for i,gtflow_path in enumerate(flow_paths):
    num = gtflow_path.split('/')[-1].strip().replace('flow.flo','img1.png')
    if not 'test' in args.dataset and not 'clip' in args.dataset:
        gtflow = read_flow(gtflow_path)
    num = num.replace('jpg','png')
    flow = read_flow('%s/%s/flo-%s'%(args.path,args.dataset,num.replace('.png', '.pfm')))
    if args.vis == 'yes':
        flowimg = flow_to_image(flow)*np.linalg.norm(flow[:,:,:2],2,2)[:,:,np.newaxis]/100./255.
        mkdir_p('%s/%s/flowimg'%(args.path,args.dataset))
        plt.imsave('%s/%s/flowimg/%s'%(args.path,args.dataset,num), flowimg)
        if 'test' in args.dataset or 'clip' in args.dataset:
            continue
        gtflowimg = flow_to_image(gtflow)*np.linalg.norm(gtflow[:,:,:2],2,2)[:,:,np.newaxis]/100./255.
        mkdir_p('%s/%s/gtimg'%(args.path,args.dataset))
        plt.imsave('%s/%s/gtimg/%s'%(args.path,args.dataset,num), gtflowimg)

    mask = gtflow[:,:,2]==1

    gtflow = gtflow[:,:,:2]
    flow = flow[:,:,:2]

    epe = np.sqrt(np.power(gtflow - flow,2).sum(-1))[mask]
    gt_mag = np.sqrt(np.power(gtflow,2).sum(-1))[mask] 

    
    clippx = [0,1000]
    inrangepx = np.logical_and((np.abs(gtflow)>=clippx[0]).sum(-1), (np.abs(gtflow)<clippx[1]).prod(-1))[mask]
    if os.path.isfile('%s/%s/%s'%(args.path,args.dataset,num.replace('png','npy'))):
        isoor = np.load('%s/%s/%s'%(args.path,args.dataset,num.replace('png','npy')))
        gtoortp = mask*((np.abs(gtflow)>clippx).sum(-1)>0)
        gtoorfp = mask*((np.abs(gtflow)>clippx).sum(-1)==0)
        oor_tp.append(isoor[gtoortp])
        oor_fp.append(isoor[gtoorfp])
    if args.vis == 'yes' and 'test' not in args.dataset:
        epeimg = np.sqrt(np.power(gtflow - flow,2).sum(-1))*(mask*(np.logical_and((np.abs(gtflow)>=clippx[0]).sum(-1), (np.abs(gtflow)<clippx[1]).prod(-1))).astype(float))
        mkdir_p('%s/%s/epeimg'%(args.path,args.dataset))
        plt.imsave('%s/%s/epeimg/%s'%(args.path,args.dataset,num), epeimg, vmax=32)

    aepe_s.append( epe[inrangepx] )
    fall_s64.append( (epe > 64)[inrangepx])
    fall_s32.append( (epe > 32)[inrangepx])
    fall_s16.append( (epe > 16)[inrangepx])
    fall_s8.append(  (epe > 8)[inrangepx])
    fall_s.append(   np.logical_and(epe > 3, epe/gt_mag > 0.05)[inrangepx])
print('flow: \t\t%.1f%%/%.3fpx'%(
                np.mean( 100*np.concatenate(fall_s,0)),
                np.mean( np.concatenate(aepe_s,0))) )
