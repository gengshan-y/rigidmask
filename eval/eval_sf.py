import os
import sys
sys.path.insert(0,os.getcwd())
import glob
import numpy as np
from matplotlib import pyplot as plt
from utils.flowlib import read_flow, flow_to_image, visualize_flow
from utils.util_flow import write_flow, readPFM
import cv2

import pdb
import PIL.Image as Image
from dataloader.exploader import load_calib_cam_to_cam, disparity_loader, triangulation
from utils.sintel_io import disparity_read, cam_read
from joblib import Parallel, delayed

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', default='/data/ptmodel/',
                    help='dataset path')
parser.add_argument('--dataset', default='2015',
                    help='dataset name')
parser.add_argument('--method', default='ours',
                    help='{ours, momodepth2}')
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
elif args.dataset == 'sinteldepth':
    from dataloader import sintel_rtn_val as DA
    datapath = '/ssd/rob_flow/training/'
elif args.dataset == 'sintelval':
    from dataloader import sintellist_val as DA
    datapath = '/ssd/rob_flow/training/'

test_left_img, test_right_img ,flow_paths= DA.dataloader(datapath)
expansionp = [i.replace('flow_occ','expansion').replace('.png', '.pfm') for i in flow_paths]
if '2015' in args.dataset:
    disp0p = [i.replace('flow_occ','disp_occ_0') for i in flow_paths]
    disp1p = [i.replace('flow_occ','disp_occ_1') for i in flow_paths]
    calib = [i.replace('flow_occ','calib_cam_to_cam').replace('_10.png', '.txt') for i in flow_paths]
elif 'sintel' in args.dataset:
    disp0p = []
    disp1p = []
    calib = []
    for fp in flow_paths:
        seqname1 = fp.split('/')[-1].split('_')[-3]
        seqname2 = fp.split('/')[-1].split('_')[-2]
        framename = int(fp.split('/')[-1].split('_')[-1].split('.')[0])
        disp0p.append('%s/disparities/%s_%s/frame_%04d.png'%(fp.rsplit('/',2)[0], seqname1, seqname2,framename+1))
        disp1p.append('%s/disparities/%s_%s/frame_%04d.png'%(fp.rsplit('/',2)[0], seqname1, seqname2,framename+2))
        calib.append('/data/gengshay/tf_depth/sintel-data/training/camdata_left/%s_%s/frame_%04d.cam'%(seqname1,seqname2,framename+1))
def eval_f(fp):
    import warnings
    warnings.filterwarnings("ignore")
    #gts
    if '2015' in args.dataset:
        gt_disp0 = disparity_loader(disp0p[fp])
        gt_disp1 = disparity_loader(disp1p[fp])
        gt_flow = read_flow(flow_paths[fp]).astype(np.float32)
        ints = load_calib_cam_to_cam(calib[fp])
        K0 = ints['K_cam2']
        fl = K0[0,0]
        bl = ints['b20']-ints['b30']
    elif 'sintel' in args.dataset:
        gt_disp0 = disparity_read(disp0p[fp])
        gt_disp1 = disparity_read(disp1p[fp])
        gt_flow = read_flow(flow_paths[fp]).astype(np.float32)
        K0,_ = cam_read(calib[fp])
        fl = K0[0,0]
        bl=0.1
    d1mask = gt_disp0>0
    d2mask = gt_disp1>0
    flowmask = gt_flow[:,:,-1]==1
    validmask = np.logical_and(np.logical_and(d1mask, d2mask), flowmask)
    if '2015' in args.dataset:
        fgmask = cv2.imread(flow_paths[fp].replace('flow_occ','obj_map'),0)>0
        fgmask = np.logical_and(fgmask,validmask)

    shape = gt_disp0.shape

    # pred
    idnum = expansionp[fp].split('/')[-1].split('.')[0]
    if args.method=='ours':
        logdc = disparity_loader( '%s/%s/mid-%s.pfm'%(args.path,args.dataset,idnum))
        pred_flow = read_flow('%s/%s/flo-%s.pfm'%(args.path,args.dataset,idnum))
    
        try:
            pred_disp = disparity_loader('%s/%s/%s_disp.pfm'%(args.path, args.dataset,idnum)) 
        except:
            try:
                pred_disp = disparity_loader('%s/%s/%s.png'%(args.path,args.dataset,idnum)) 
            except:
                try:
                    pred_disp = disparity_loader('%s/%s/disp-%s.pfm'%(args.path, args.dataset,idnum)) 
                except:
                    pred_disp = disparity_loader('%s/%s/exp-%s.pfm'%(args.path,args.dataset,idnum)) 
        pred_disp[pred_disp==np.inf]=pred_disp[pred_disp!=np.inf].max()
        pred_disp[np.isnan(pred_disp)]=1e-12
        pred_disp[pred_disp<1e-12]=1e-12    

        pred_disp1= pred_disp/np.exp(logdc)
        pred_flow = disparity_loader( '%s/%s/flo-%s.pfm'%(args.path,args.dataset,idnum))
    elif args.method=='monodepth2':
        pred_disp = disparity_loader( '%s/%s/%s_disp.pfm'%(args.path,args.dataset,idnum))
    else:exit()    

    #hom_p = np.stack((pred_disp.flatten(), np.ones(pred_disp.flatten().shape))).T[validmask.flatten()]
    #xx = np.linalg.inv(np.matmul(hom_p[:,:,np.newaxis],hom_p[:,np.newaxis,:]).sum(0))
    #yy = (hom_p[:,:,np.newaxis]*gt_disp0.flatten()[validmask.flatten(),np.newaxis,np.newaxis]).sum(0)
    #st = xx.dot(yy)
    #pred_disp  = pred_disp*st[0] + st[1]

    scale_factor = np.median((gt_disp0/pred_disp)[validmask])
    pred_disp  = scale_factor*pred_disp
    pred_disp1 = scale_factor*pred_disp1

    # eval
    d1err = np.abs(pred_disp-gt_disp0)
    d1err_map = (np.logical_and(d1err>=3, d1err/gt_disp0>=0.05))
    d1err = d1err_map[validmask]

    d2err = np.abs(pred_disp1-gt_disp1)
    d2err_map = (np.logical_and(d2err>=3, d2err/gt_disp1>=0.05))
    d2err = d2err_map[validmask]

    flow_epe = np.sqrt(np.power(gt_flow - pred_flow,2).sum(-1))
    gt_flow_mag = np.linalg.norm(gt_flow[:,:,:2],2,-1)
    flerr_map = np.logical_and(flow_epe > 3, flow_epe/gt_flow_mag > 0.05)
    flerr = flerr_map[validmask]
    flerr_map[~validmask]=False

    try:
        d1ferr = d1err_map[fgmask]
        d2ferr = d2err_map[fgmask]
        flferr = flerr_map[fgmask]
    except: 
        d1ferr=np.zeros(1)
        d2ferr=np.zeros(1)
        flferr=np.zeros(1)

    sferr = np.logical_or(np.logical_or(d1err,d2err),flerr)
    sfferr = np.logical_or(np.logical_or(d1ferr,d2ferr),flferr)

    img = cv2.imread(test_left_img[fp])[:,:,::-1]
#    cv2.imwrite('%s/%s/err-%s.png'%(args.path,args.dataset,idnum),np.vstack((gt_disp0,pred_disp,gt_disp1,pred_disp1,flerr_map.astype(float)*255)))
    if '2015' in args.dataset:
        flowvis = cv2.imread(test_left_img[fp].replace('image_2', 'viz_flow_occ'))[:,:,::-1]
    else:
        flowvis = flow_to_image(gt_flow)
    pred_flow[:,:,-1]=1
    cv2.imwrite('%s/%s/err-%s.png'%(args.path,args.dataset,idnum),np.vstack((img, flowvis,255*visualize_flow(pred_flow,mode='RGB'),np.tile(flerr_map.reshape(shape)[:,:,None],3).astype(float)*255))[:,:,::-1])
    return d1err.mean(), d2err.mean(), flerr.mean(), sferr.mean(),\
           d1ferr.mean(), d2ferr.mean(), flferr.mean(), sfferr.mean(),gt_flow_mag.mean() 

# for sintel, only evaluate frames with GT flow magnitude greater than 5px.    
if 'sintel' in args.dataset:
    eval_th = 5
else:
    eval_th = 0
rt = Parallel(n_jobs=8)(delayed(eval_f)(fp) for fp in range(len(test_left_img)))
d1_error = [k[0] for k in rt                 if k[8]>eval_th]
d2_error = [k[1] for k in rt                 if k[8]>eval_th]
fl_error = [k[2] for k in rt                 if k[8]>eval_th]
sf_error = [k[3] for k in rt                 if k[8]>eval_th]
d1f_error = [k[4] for k in rt                if k[8]>eval_th]
d2f_error = [k[5] for k in rt                if k[8]>eval_th]
flf_error = [k[6] for k in rt                if k[8]>eval_th]
sff_error = [k[7] for k in rt                if k[8]>eval_th]
disp0p = [k for i,k in enumerate(disp0p) if rt[i][8]>eval_th]

if args.verbose:
    for i in range(len(d1_error)):
        print('%.2f/%.2f/%.2f/%s'%(100*d1_error[i],100*flf_error[i],100*sf_error[i], disp0p[i]))
print('%d images'%len(d1_error))
print('d1-err (a/f):\t%.2f%%\t%.2f%%'%(100*np.mean(d1_error),100*np.mean(d1f_error)))
print('d2-err (a/f):\t%.2f%%\t%.2f%%'%(100*np.mean(d2_error),100*np.mean(d2f_error)))
print('fl-err (a/f):\t%.2f%%\t%.2f%%'%(100*np.mean(fl_error),100*np.mean(flf_error)))
print('sf-err (a/f):\t%.2f%%\t%.2f%%'%(100*np.mean(sf_error),100*np.mean(sff_error)))
