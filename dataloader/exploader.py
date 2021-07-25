import os
import numbers
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
import torchvision
from . import exp_transforms
import pdb
import cv2
from utils.flowlib import read_flow
from utils.util_flow import readPFM, load_calib_cam_to_cam
from utils.sintel_io import disparity_read, cam_read

def default_loader(path):
    return Image.open(path).convert('RGB')

def flow_loader(path):
    if '.pfm' in path:
        data =  readPFM(path)[0]
        data[:,:,2] = 1
        return data
    else:
        return read_flow(path)

def load_exts(cam_file):
    with open(cam_file, 'r') as f:
        lines = f.readlines()

    l_exts = []
    r_exts = []
    for l in lines:
        if 'L ' in l:
            l_exts.append(np.asarray([float(i) for i in l[2:].strip().split(' ')]).reshape(4,4))
        if 'R ' in l:
            r_exts.append(np.asarray([float(i) for i in l[2:].strip().split(' ')]).reshape(4,4))
    return l_exts,r_exts        

def disparity_loader(path):
    if '.png' in path:
        data = Image.open(path)
        data = np.ascontiguousarray(data,dtype=np.float32)/256
        return data
    else:    
        return readPFM(path)[0]

# triangulation
def triangulation(disp, xcoord, ycoord, bl=1, fl = 450, cx = 479.5, cy = 269.5):
    depth = bl*fl / disp # 450px->15mm focal length
    X = (xcoord - cx) * depth / fl
    Y = (ycoord - cy) * depth / fl
    Z = depth
    P = np.concatenate((X[np.newaxis],Y[np.newaxis],Z[np.newaxis]),0).reshape(3,-1)
    P = np.concatenate((P,np.ones((1,P.shape[-1]))),0)
    return P

class myImageFloder(data.Dataset):
    def __init__(self, iml0, iml1, flowl0, loader=default_loader, dploader= flow_loader, scale=1.,shape=[320,448], order=1, noise=0.06, pca_augmentor=True, prob = 1.,disp0=None,disp1=None,calib=None ):
        self.iml0 = iml0
        self.iml1 = iml1
        self.flowl0 = flowl0
        self.loader = loader
        self.dploader = dploader
        self.scale=scale
        self.shape=shape
        self.order=order
        self.noise = noise
        self.pca_augmentor = pca_augmentor
        self.prob = prob
        self.disp0 = disp0
        self.disp1 = disp1
        self.calib = calib

    def __getitem__(self, index):
        iml0  = self.iml0[index]
        iml1 = self.iml1[index]
        flowl0= self.flowl0[index]
        th, tw = self.shape

        iml0 = self.loader(iml0)
        iml1 = self.loader(iml1)

        # get disparity
        flowl0 = self.dploader(flowl0)
        flowl0[:,:,-1][flowl0[:,:,0]==np.inf]=0  # for gtav window pfm files
        flowl0[:,:,0][~flowl0[:,:,2].astype(bool)]=0
        flowl0[:,:,1][~flowl0[:,:,2].astype(bool)]=0  # avoid nan in grad
        flowl0 = np.ascontiguousarray(flowl0,dtype=np.float32)
        flowl0[np.isnan(flowl0)] = 1e6 # set to max
        if 'camera_data.txt' in self.calib[index]: # synthetic scene flow
            bl=1
            if '15mm_' in self.calib[index]: 
                fl=450 # 450
            else:
                fl=1050
            cx = 479.5
            cy = 269.5
            # negative disp
            d1 = np.abs(disparity_loader(self.disp0[index]))
            d2 = np.abs(disparity_loader(self.disp1[index]) + d1)
        elif 'Sintel' in self.calib[index]:
            K0,_ = cam_read(self.calib[index])
            fl = K0[0,0]
            bl = 0.1
            cx=K0[0,2]
            cy=K0[1,2]
            d1 = disparity_read(self.disp0[index])
            d2_nf = disparity_read(self.disp1[index])

            shape = d1.shape
            x0,y0=np.meshgrid(range(shape[1]),range(shape[0]))
            x0=x0.astype(np.float32)
            y0=y0.astype(np.float32)
            x1=x0+flowl0[:,:,0]
            y1=y0+flowl0[:,:,1]
            d2 = cv2.remap(d2_nf,x1,y1,cv2.INTER_LINEAR)
            re_iml1 = cv2.remap(np.asarray(iml1),x1,y1,cv2.INTER_LINEAR)
            d2[np.logical_or(d2<0, np.linalg.norm(np.asarray(iml0)-re_iml1,axis=2)>50)]=0
        elif self.calib[index] == 'udf': # undefined dataset
            fl = 1
            cx = 0
            cy = 0
            bl = 1
            d1 = 100./disparity_loader(self.disp0[index])
            d2_nf = 100./disparity_loader(self.disp1[index])
            shape = d1.shape
            x0,y0=np.meshgrid(range(shape[1]),range(shape[0]))
            x0=x0.astype(np.float32)
            y0=y0.astype(np.float32)
            x1=x0+flowl0[:,:,0]
            y1=y0+flowl0[:,:,1]
            d2 = cv2.remap(d2_nf,x1,y1,cv2.INTER_LINEAR)
            re_iml1 = cv2.remap(np.asarray(iml1),x1,y1,cv2.INTER_LINEAR)
            d2[np.logical_or(d2<0, np.linalg.norm(np.asarray(iml0)-re_iml1,axis=2)>50)]=0
            cv2.imwrite('/data/gengshay/0.png', d1)
            cv2.imwrite('/data/gengshay/1.png', d2)
        else: # kitti
            ints = load_calib_cam_to_cam(self.calib[index])
            fl = ints['K_cam2'][0,0]
            cx = ints['K_cam2'][0,2]
            cy = ints['K_cam2'][1,2]
            bl = ints['b20']-ints['b30']
            d1 = disparity_loader(self.disp0[index])
            d2 = disparity_loader(self.disp1[index])
        #flowl0[:,:,2] = (flowl0[:,:,2]==1).astype(float)
        flowl0[:,:,2] = np.logical_and(np.logical_and(flowl0[:,:,2]==1, d1!=0), d2!=0).astype(float)

        shape = d1.shape
        mesh = np.meshgrid(range(shape[1]),range(shape[0]))
        xcoord = mesh[0].astype(float)
        ycoord = mesh[1].astype(float)
        
        # triangulation in two frames
        P0 = triangulation(d1, xcoord, ycoord, bl=bl, fl = fl, cx = cx, cy = cy)
        P1 = triangulation(d2, xcoord + flowl0[:,:,0], ycoord + flowl0[:,:,1], bl=bl, fl = fl, cx = cx, cy = cy)
        depth0 = P0[2]
        depth1 = P1[2]

        # first frame depth and 3d flow
        depth0 =  depth0.reshape(shape).astype(np.float32)
        flow3d = (P1-P0)[:3].reshape((3,)+shape).transpose((1,2,0))

        # add rectified 3D flow
        if ('_R.pfm' in self.flowl0[index]) or ('_L.pfm' in self.flowl0[index]): # synthetic scene flow
            # from https://github.com/Wallacoloo/printipi
            fid = int(self.flowl0[index].split('/')[-1].split('_')[1])
            with open(self.calib[index],'r') as f:
                fid = fid - int(f.readline().split(' ')[-1])
            # extrinsics
            l_exts,r_exts= load_exts(self.calib[index])
            if '/right/' in self.iml0[index]:
                exts = r_exts
            else:
                exts = l_exts
            if '/into_future/' in self.flowl0[index]:
                if (fid+1)>len(exts)-1: print(self.flowl0[index])
                if (fid)>len(exts)-1: print(self.flowl0[index])
                ext1 = exts[fid+1]
                ext0 = exts[fid]
            else:
                if (fid-1)>len(exts)-1: print(self.flowl0[index])
                if (fid)>len(exts)-1: print(self.flowl0[index])
                ext1 = exts[fid-1]
                ext0 = exts[fid]
            camT = np.eye(4); camT[1,1]=-1; camT[2,2]=-1
            RT01 = camT.dot(np.linalg.inv(ext0)).dot(ext1).dot(camT)
        elif 'kitti_scene' in self.iml0[index]: # kitti:
            RT01 = np.loadtxt(self.iml0[index].replace('image_2', 'pose').replace('.png', '.txt'))
        elif 'info.pkl' in self.calib[index]: # refresh:
            fid = int(self.flowl0[index].split('/')[-1].split('.')[0])
            RT01 = np.linalg.inv(infofile['pose'][fid]).dot(infofile['pose'][fid-1])  # backward camera motion
        else:
            RT01 = np.eye(4); RT01[2,-1] = 1.
        rect_flow3d = (RT01[:3,:3].dot(P1[:3])-P0[:3]).reshape((3,)+shape).transpose((1,2,0))

        depthflow = np.concatenate((depth0[:,:,np.newaxis],rect_flow3d,flow3d),2)
        RT01 = np.concatenate((cv2.Rodrigues(RT01[:3,:3])[0][:,0],RT01[:3,-1])).astype(np.float32)
        
        # append obj mask
        if 'kitti_scene' in self.iml0[index]:
            fnum = int(self.iml0[index].split('/')[-1].split('_')[0])
            obj_fname = self.iml0[index].replace('image_2', 'obj_map')
            #obj_fname = '/home/gengshay/moseg-detectron2/%06d.png'%(fnum)
            obj_idx = cv2.imread(obj_fname,0)
        else:
            fnum = int(self.iml0[index].split('/')[-1].split('.png')[0])
            obj_fname = '%s/%04d.pfm'%(self.flowl0[index].replace('/optical_flow','object_index').replace('into_past/','/').replace('into_future/','/').rsplit('/',1)[0],fnum)
            obj_idx = disparity_loader(obj_fname)
        depthflow = np.concatenate((depthflow,obj_idx[:,:,np.newaxis]),2)

        iml1 = np.asarray(iml1)/255.
        iml0 = np.asarray(iml0)/255.
        iml0 = iml0[:,:,::-1].copy()
        iml1 = iml1[:,:,::-1].copy()

        ## following data augmentation procedure in PWCNet 
        ## https://github.com/lmb-freiburg/flownet2/blob/master/src/caffe/layers/data_augmentation_layer.cu
        import __main__ # a workaround for "discount_coeff"
        try:
            with open('/scratch/gengshay/iter_counts-%d.txt'%int(__main__.args.logname.split('-')[-1]), 'r') as f:
                iter_counts = int(f.readline())
        except:
            iter_counts = 0
        schedule = [0.5, 1., 50000.]  # initial coeff, final_coeff, half life
        schedule_coeff = schedule[0] + (schedule[1] - schedule[0]) * \
          (2/(1+np.exp(-1.0986*iter_counts/schedule[2])) - 1)

        if self.pca_augmentor:
            pca_augmentor = exp_transforms.pseudoPCAAug( schedule_coeff=schedule_coeff)
        else:
            pca_augmentor = exp_transforms.Scale(1., order=0)

        if np.random.binomial(1,self.prob):
            co_transform1 = exp_transforms.Compose([
                           exp_transforms.SpatialAug([th,tw],
                                           scale=[0.2,0.,0.1],
                                           rot=[0.4,0.],
                                           trans=[0.4,0.],
                                           squeeze=[0.3,0.], schedule_coeff=schedule_coeff, order=self.order),
            ])
        else:
            co_transform1 = exp_transforms.Compose([
            exp_transforms.RandomCrop([th,tw]),
            ])

        co_transform2 = exp_transforms.Compose([
            exp_transforms.pseudoPCAAug( schedule_coeff=schedule_coeff),
            #exp_transforms.PCAAug(schedule_coeff=schedule_coeff),
            exp_transforms.ChromaticAug( schedule_coeff=schedule_coeff, noise=self.noise),
            ])

        flowl0 = np.concatenate([flowl0,depthflow],-1)
        augmented,flowl0,intr = co_transform1([iml0, iml1], flowl0, [fl,cx,cy,bl])
        imol0 = augmented[0]
        imol1 = augmented[1]
        augmented,flowl0,intr = co_transform2(augmented, flowl0, intr)

        iml0 = augmented[0]
        iml1 = augmented[1]
        flowl0 = flowl0.astype(np.float32)
        depthflow = flowl0[:,:,3:]
        flowl0 = flowl0[:,:,:3]

        # randomly cover a region
        sx=0;sy=0;cx=0;cy=0
        if np.random.binomial(1,0.5):
            sx = int(np.random.uniform(25,100))
            sy = int(np.random.uniform(25,100))
            #sx = int(np.random.uniform(50,150))
            #sy = int(np.random.uniform(50,150))
            cx = int(np.random.uniform(sx,iml1.shape[0]-sx))
            cy = int(np.random.uniform(sy,iml1.shape[1]-sy))
            iml1[cx-sx:cx+sx,cy-sy:cy+sy] = np.mean(np.mean(iml1,0),0)[np.newaxis,np.newaxis]

        iml0  = torch.Tensor(np.transpose(iml0,(2,0,1)))
        iml1  = torch.Tensor(np.transpose(iml1,(2,0,1)))
        imol0 = imol0[:,:,::-1].copy()  # RGB
        imol1 = imol1[:,:,::-1].copy()

        return iml0, iml1, flowl0, depthflow, intr, imol0, imol1, np.asarray([cx-sx,cx+sx,cy-sy,cy+sy]),RT01

    def __len__(self):
        return len(self.iml0)
