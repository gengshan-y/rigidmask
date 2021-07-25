import argparse
import collections
import cv2
import numpy as np
import os
import pdb
import random
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
torch.backends.cudnn.benchmark=True
autocast = torch.cuda.amp.autocast

from utils.flowlib import flow_to_image
from utils.io import add_image
from models import *
from utils.multiscaleloss import realEPE

parser = argparse.ArgumentParser(description='VCNPlus')
parser.add_argument('--maxdisp', type=int ,default=256,
                    help='maxium disparity, out of range pixels will be masked out. Only affect the coarsest cost volume size (default 256)')
parser.add_argument('--fac', type=float ,default=1,
                    help='controls the shape of search grid. Only affect the coarsest cost volume size (default 1)')
parser.add_argument('--logname', default='exp-1',
                    help='name of the log file (default exp-1)')
parser.add_argument('--database',
                    help='path to the database (required)')
parser.add_argument('--loadmodel', default=None,
                    help='path of the pre-trained model (default None)')
parser.add_argument('--loadflow', default=None,
                    help='path of the pre-trained flow model (default None)')
parser.add_argument('--savemodel',
                    help='path to save the model (required)')
parser.add_argument('--retrain', default='true',
                    help='whether to reset moving mean / other hyperparameters (default true)')
parser.add_argument('--stage', default='expansion',
                    help='one of {chairs, things, 2015train, 2015trainval, sinteltrain, sinteltrainval, expansion, expansion2015train, expansion2015tv} (deafult expansion)')
parser.add_argument('--nproc', type=int, default=1,
                    help='number of process to use (default 1)')
parser.add_argument('--ngpus', type=int, default=1,
                    help='(deprecated) number of gpus to use before ddp (default 1)')
parser.add_argument('--itersave', default='./',
                    help='a dir to save iteration counts (default ./)')
parser.add_argument('--niter', type=int ,default=300000,
                    help='maximum iteration (default 300k)')
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()

# distributed dataparallel
torch.cuda.set_device(args.local_rank)
world_size = args.nproc
torch.distributed.init_process_group(
    'nccl',
    init_method='env://',
    world_size=world_size,
    rank=args.local_rank,
)

# fix random seed
torch.manual_seed(1)
def _init_fn(worker_id):
    np.random.seed()
    random.seed()
torch.manual_seed(8)  # do it again
torch.cuda.manual_seed(1)

## set hyperparameters for training
ngpus = args.ngpus
worker_mul = int(1)

if 'expansion' in args.stage:
    datashape = [320,640]
    batch_size = 6*ngpus
elif 'seg' in args.stage:
    datashape = [320,640]
    batch_size = 6*ngpus
elif args.stage == 'chairs' or args.stage == 'things':
    datashape = [320,448]
    batch_size = 4*ngpus
elif '2015' in args.stage:
    datashape = [256,768]
    batch_size = 4*ngpus
elif 'sintel' in args.stage:
    datashape = [320,576]
    batch_size = 4*ngpus
elif args.stage == 'rob':
    datashape = [320,640]
    batch_size = 3*ngpus
else: 
    print('error')
    exit(0)

## dataloader
## expansion datasets
if 'expansion' in args.stage:
    from dataloader import exploader as dd
    if '2015' in args.stage:
        if 'train' in args.stage:
            from dataloader import kitti15list_train as lk15
        elif 'tv' in args.stage:
            from dataloader import kitti15list as lk15
        iml0, iml1, flowl0 = lk15.dataloader('%s/kitti_scene/training/'%args.database)
        disp0 = [i.replace('flow_occ','disp_occ_0') for i in flowl0]
        disp1 = [i.replace('flow_occ','disp_occ_1') for i in flowl0]
        calib = [i.replace('flow_occ','calib')[:-7]+'.txt' for i in flowl0]
        loader_kitti15_sc = dd.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=0,prob=0.5,disp0=disp0, disp1=disp1, calib=calib)
    else:
        from dataloader import sceneflowlist as lsf
        iml0, iml1, flowl0, disp0, dispc, calib = lsf.dataloader('%s/Driving/'%args.database, level=6)
        loader_driving_sc = dd.myImageFloder(iml0,iml1,flowl0, shape=datashape, disp0=disp0,disp1=dispc,calib=calib)
        iml0, iml1, flowl0, disp0, dispc, calib = lsf.dataloader('%s/Monkaa/'%args.database, level=4)
        loader_monkaa_sc = dd.myImageFloder(iml0,iml1,flowl0, shape=datashape, disp0=disp0,disp1=dispc,calib=calib)
        iml0, iml1, flowl0, disp0, dispc, calib = lsf.dataloader('/ssd1/gengshay/FlyingThings3D/', level=6)
        loader_things_sc = dd.myImageFloder(iml0,iml1,flowl0, shape=datashape, disp0=disp0,disp1=dispc,calib=calib)
        # kitti
        from dataloader import kitti15list_train as lk15
        iml0, iml1, flowl0 = lk15.dataloader('%s/kitti_scene/training/'%args.database)
        disp0 = [i.replace('flow_occ','disp_occ_0') for i in flowl0]
        disp1 = [i.replace('flow_occ','disp_occ_1') for i in flowl0]
        calib = [i.replace('flow_occ','calib')[:-7]+'.txt' for i in flowl0]
        loader_kitti15_sc = dd.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=0,prob=0.5,disp0=disp0, disp1=disp1, calib=calib)
        # sintel
        from dataloader import sintellist_train as ls
        iml0, iml1, flowl0 = ls.dataloader('%s/rob_flow/training/'%args.database)
        disp0 = []; disp1 = []; calib = []
        for impath in iml0:
            passname = impath.split('/')[-1].split('_')[-4]
            seqname1 = impath.split('/')[-1].split('_')[-3]
            seqname2 = impath.split('/')[-1].split('_')[-2]
            framename = int(impath.split('/')[-1].split('_')[-1].split('.')[0])
            disp0.append('%s/Sintel/disparities/%s_%s/frame_%04d.png'%(impath.rsplit('/',2)[0], seqname1, seqname2,framename+1))
            disp1.append('%s/Sintel/disparities/%s_%s/frame_%04d.png'%(impath.rsplit('/',2)[0], seqname1, seqname2,framename+2))
            calib.append('%s/Sintel/camdata_left/%s_%s/frame_%04d.cam'%(impath.rsplit('/',2)[0], seqname1,seqname2,framename+1))
        loader_sintel_sc = dd.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=1, noise=0, disp0=disp0, disp1=disp1, calib=calib)
elif 'seg' in args.stage:
    from dataloader import exploader as dd
    from dataloader import sceneflowlist as lsf
    iml0, iml1, flowl0, disp0, dispc, calib = lsf.dataloader('%s/Driving/'%args.database, level=6)
    loader_driving_sc = dd.myImageFloder(iml0,iml1,flowl0, shape=datashape, disp0=disp0,disp1=dispc,calib=calib)
    iml0, iml1, flowl0, disp0, dispc, calib = lsf.dataloader('%s/Monkaa/'%args.database, level=4)
    loader_monkaa_sc = dd.myImageFloder(iml0,iml1,flowl0, shape=datashape, disp0=disp0,disp1=dispc,calib=calib)
    iml0, iml1, flowl0, disp0, dispc, calib = lsf.dataloader('%s/FlyingThings3D/', level=6)
    loader_things_sc = dd.myImageFloder(iml0,iml1,flowl0, shape=datashape, disp0=disp0,disp1=dispc,calib=calib)
    # kitti
    from dataloader import kitti15list as lk15
    iml0, iml1, flowl0 = lk15.dataloader('%s/kitti_scene/training/'%args.database)
    disp0 = [i.replace('flow_occ','disp_occ_0') for i in flowl0]
    disp1 = [i.replace('flow_occ','disp_occ_1') for i in flowl0]
    calib = [i.replace('flow_occ','calib')[:-7]+'.txt' for i in flowl0]
    # dense disp
    disp0 = [i.replace('disp_occ_0','disp_occ_0_ganet') for i in disp0]
    loader_kitti15_sc = dd.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=0,prob=0.5,disp0=disp0, disp1=disp1, calib=calib)
else: # flow
    from dataloader import robloader as dr
    if args.stage == 'chairs' or 'sintel' in args.stage or args.stage=='rob':
        # flying chairs
        from dataloader import chairslist as lc
        iml0, iml1, flowl0 = lc.dataloader('%s/FlyingChairs_release/data/'%args.database)
        with open('misc/order.txt','r') as f:
            order = [int(i) for i in f.readline().split(' ')]
        with open('misc/FlyingChairs_train_val.txt', 'r') as f:
            split = [int(i) for i in f.readlines()]
        iml0 = [iml0[i] for i in order     if split[i]==1]
        iml1 = [iml1[i] for i in order     if split[i]==1]
        flowl0 = [flowl0[i] for i in order if split[i]==1]
        loader_chairs = dr.myImageFloder(iml0,iml1,flowl0, shape = datashape)
    if args.stage == 'things' or 'sintel' in args.stage or args.stage=='rob':
        # flything things
        from dataloader import thingslist as lt
        iml0, iml1, flowl0 = lt.dataloader('/ssd0/gengshay/FlyingThings3D_subset/train/')
        loader_things = dr.myImageFloder(iml0,iml1,flowl0,shape = datashape,scale=1, order=1)
    
    # fine-tuning datasets
    if args.stage == '2015train' or args.stage=='rob':
        from dataloader import kitti15list_train as lk15
    else:
        from dataloader import kitti15list as lk15
    if args.stage == 'sinteltrain' or args.stage=='rob':
        from dataloader import sintellist_train as ls
    else:
        from dataloader import sintellist as ls
    from dataloader import kitti12list as lk12
    from dataloader import hd1klist_train as lh
    
    if 'sintel' in args.stage:
        iml0, iml1, flowl0 = lk15.dataloader('%s/kitti_scene/training/'%args.database)
        loader_kitti15 = dr.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=0, noise=0)  # SINTEL
        iml0, iml1, flowl0 = lh.dataloader('%s/rob_flow/training/'%args.database)
        loader_hd1k = dr.myImageFloder(iml0,iml1,flowl0,shape=datashape, scale=0.5,order=0, noise=0)
        iml0, iml1, flowl0 = ls.dataloader('%s/rob_flow/training/'%args.database)
        loader_sintel = dr.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=1, noise=0)
        #loader_sintel = dr.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=1, noise=0, scale_aug=[0.2,0.])
    if '2015' in args.stage:
        iml0, iml1, flowl0 = lk12.dataloader('%s/data_stereo_flow/training/'%args.database)
        #loader_kitti12 = dr.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=0, prob=0.5, scale_aug=[0.2,0.])
        loader_kitti12 = dr.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=0, prob=0.5)
        iml0, iml1, flowl0 = lk15.dataloader('%s/kitti_scene/training/'%args.database)
        #loader_kitti15 = dr.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=0, prob=0.5, scale_aug=[0.2,0.])  # KITTI
        loader_kitti15 = dr.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=0, prob=0.5)  # KITTI
    if args.stage=='rob':
        #from dataloader import kitti15list as lk15
        #from dataloader import sintellist as ls
        #from dataloader import viperlist as lv
        from dataloader import kitti15list_train as lk15
        from dataloader import sintellist_train as ls
        from dataloader import viperlist_train as lv
        from dataloader import hd1klist as lh
        iml0, iml1, flowl0 = lk12.dataloader('%s/data_stereo_flow/training/'%args.database)
        loader_kitti12 = dr.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=0, prob=0.5)
        iml0, iml1, flowl0 = lk15.dataloader('%s/kitti_scene/training/'%args.database)
        loader_kitti15 = dr.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=0, prob=0.5)  # KITTI
        iml0, iml1, flowl0 = lh.dataloader('%s/rob_flow/training/'%args.database)
        loader_hd1k = dr.myImageFloder(iml0,iml1,flowl0,shape=datashape, scale=0.5,order=1, noise=0)
        iml0, iml1, flowl0 = ls.dataloader('%s/rob_flow/training/'%args.database)
        loader_sintel = dr.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=1, noise=0)
        from dataloader import sceneflowlist as lsf
        iml0, iml1, flowl0, disp0, dispc, calib = lsf.dataloader('%s/Driving/'%args.database, level=6)
        loader_driving = dr.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=1)
        iml0, iml1, flowl0, disp0, dispc, calib = lsf.dataloader('%s/Monkaa/'%args.database, level=4)
        loader_monkaa = dr.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=1)
        iml0, iml1, flowl0 = lv.dataloader('%s/rob_flow/training/'%args.database)
        loader_viper = dr.myImageFloder(iml0,iml1,flowl0, shape = datashape, scale=1, order=1, scale_aug=[0.8,-0.2])

## aggregate datasets
if 'expansion' in args.stage:
    if '2015' in args.stage:
        data_inuse = torch.utils.data.ConcatDataset([loader_kitti15_sc]*10000)
    else:
        #data_inuse = torch.utils.data.ConcatDataset([loader_driving_sc]*200+[loader_monkaa_sc]*100+[loader_things_sc]*40 + [loader_kitti15_sc]*22000 + [loader_sintel_sc]*2200)
        #data_inuse = torch.utils.data.ConcatDataset([loader_driving_sc]*200+[loader_monkaa_sc]*100+[loader_things_sc]*40 + [loader_gtav_sc]*700)  # no kitti sintel
        data_inuse = torch.utils.data.ConcatDataset([loader_driving_sc]*200+[loader_monkaa_sc]*100+[loader_things_sc]*40)  # no kitti sintel
    for i in data_inuse.datasets:
        i.black = False
        i.cover = True
    baselr = 1e-3
    num_steps = 7e4
elif 'seg' in args.stage:
    if args.stage=='segsf':
        #data_inuse = torch.utils.data.ConcatDataset([loader_driving_sc]*200+[loader_things_sc]*40)
        data_inuse = torch.utils.data.ConcatDataset([loader_driving_sc]*200+[loader_monkaa_sc]*100+[loader_things_sc]*40)
    elif args.stage=='segkitti':
        #data_inuse = torch.utils.data.ConcatDataset([loader_kitti15_sc]*22000)
        #data_inuse = torch.utils.data.ConcatDataset([loader_driving_sc]*200+[loader_things_sc]*40 + [loader_kitti15_sc]*22000)
        data_inuse = torch.utils.data.ConcatDataset([loader_driving_sc]*200+[loader_monkaa_sc]*100+[loader_things_sc]*40 + [loader_kitti15_sc]*22000)
    for i in data_inuse.datasets:
        i.black = False
        i.cover = True
    baselr=5e-4
    num_steps = 7e4
elif args.stage=='chairs':
    data_inuse = torch.utils.data.ConcatDataset([loader_chairs]*100) 
    baselr = 1e-3
    num_steps = 7e4
elif args.stage=='things':
    data_inuse = torch.utils.data.ConcatDataset([loader_things]*100) 
    baselr = 1e-3
    num_steps = 7e4
elif '2015' in args.stage:
    data_inuse = torch.utils.data.ConcatDataset([loader_kitti15]*50+[loader_kitti12]*50)
    for i in data_inuse.datasets:
        i.black = True
        i.cover = True
elif 'sintel' in args.stage:
    data_inuse = torch.utils.data.ConcatDataset([loader_kitti15]*200*6+[loader_hd1k]*40*6 + [loader_sintel]*150*6 + [loader_chairs]*2*6 + [loader_things]*6)
    for i in data_inuse.datasets:
        i.black = True
        i.cover = True
    baselr = 1e-4
elif args.stage=='rob':
    data_inuse = torch.utils.data.ConcatDataset([loader_kitti12]*2700+[loader_kitti15]*2700 + [loader_sintel]*600 + [loader_chairs]*12 + [loader_things]*6 + [loader_hd1k]*900 + [loader_driving]*50 + [loader_monkaa]*25+[loader_viper]*70)
    #data_inuse = torch.utils.data.ConcatDataset([loader_chairs]*12 + [loader_things]*6 + [loader_driving]*50 + [loader_monkaa]*25+ [loader_viper]*70)  # noks
    for i in data_inuse.datasets:
        i.black = True
        i.cover = True
    baselr = 1e-3
    num_steps = 7e4
else:
    print('error')
    exit(0)


print('Total iterations: %d'%(len(data_inuse)//batch_size))
print('Max iterations: %d'  %(args.niter))

from models.VCNplus  import VCN 
model = VCN([batch_size//ngpus]+data_inuse.datasets[0].shape[::-1], 
md=[int(4*(args.maxdisp/256)), 4,4,4,4], fac=args.fac, exp_unc= args.loadmodel is None or not ('kitti' in args.loadmodel))

# sync bn and dataparallel
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
device = torch.device('cuda:{}'.format(args.local_rank))
model = model.to(device)
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[args.local_rank],
    output_device=args.local_rank,
    find_unused_parameters=True
)


total_iters = 0
mean_L=[[0.33,0.33,0.33]]
mean_R=[[0.33,0.33,0.33]]
if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel,map_location='cpu')
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items()}
    #pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'fgnet' not in k and 'det' not in k}
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)

    if args.retrain == 'true':
        print('re-training')
        if 'expansion' in args.stage or 'depth' in args.stage or 'seg' in args.stage:
            print('resuming mean from %d'%total_iters)
            mean_L=pretrained_dict['mean_L']
            mean_R=pretrained_dict['mean_R']
    else:
        with open('%s/iter_counts-%d.txt'%(args.itersave, int(args.logname.split('-')[-1])), 'r') as f:
            total_iters = int(f.readline())
        print('resuming from %d'%total_iters)
        mean_L=pretrained_dict['mean_L']
        mean_R=pretrained_dict['mean_R']

if args.loadflow is not None:
    pretrained_dict = torch.load(args.loadflow,map_location='cpu')
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'f_modules' in k or 'p_modules' in k or 'oor_modules' in k or 'fuse_modules' in k}
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)


mix_precision = False
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
optimizer = optim.AdamW(model.parameters(), lr=baselr, weight_decay=0.0001, eps=1e-8)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, baselr, int(num_steps+100),
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
if args.local_rank==0: log = SummaryWriter('%s/%s'%(args.savemodel,args.logname), comment = args.logname)
scaler = GradScaler(enabled=mix_precision)

def train(imgL,imgR,flowl0,imgAux,intr, imgoL, imgoR, occp, RT01):
        model.train()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   
        flowl0 = Variable(torch.FloatTensor(flowl0))

        imgL, imgR, flowl0 = imgL.cuda(device), imgR.cuda(device), flowl0.cuda(device)
        # mask: valid flow GT & within pre-defined range 
        mask = (flowl0[:,:,:,2] == 1) & (flowl0[:,:,:,0].abs() < args.maxdisp) & (flowl0[:,:,:,1].abs() < (args.maxdisp//args.fac))
        if not imgAux is None:
            imgAux = imgAux.cuda(device)
            imgoL, imgoR = imgoL.float().cuda(device), imgoR.float().cuda(device)
            # mask: + 0.01<depth<100, imgAux: depth, d1,d2,d2,flow3d
            mask = mask & (imgAux[:,:,:,0] < 100) & (imgAux[:,:,:,0] > 0.01) 
            exp_flag = True
        else:
            exp_flag = False
        if 'expansion' in args.stage:
            exp_flag = 1 # expanson
        elif 'seg' in args.stage:
            exp_flag = 2 # segmentation
        else:
            exp_flag = 0 # flow
        mask.detach_(); 

        # rearrange inputs
        groups = []
        for i in range(ngpus):
            groups.append(imgL[i*batch_size//ngpus:(i+1)*batch_size//ngpus])
            groups.append(imgR[i*batch_size//ngpus:(i+1)*batch_size//ngpus])

        # forward-backward
        optimizer.zero_grad()
        disp_input = None
        #disp_input = 1./torch.clamp(imgAux[:,:,:,0],1,100)[:,np.newaxis]
        with autocast(enabled=mix_precision):
            output = model(torch.cat(groups,0), [flowl0,mask,imgAux,intr, imgoL, imgoR, occp, RT01, exp_flag],disp_input=disp_input)
            loss = output[-3].mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)                
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
        scaler.step(optimizer)
        scheduler.step()
#        for param_group in optimizer.param_groups:
#            print(param_group['lr'])
        scaler.update()

#        loss.backward()
#        #torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
#        optimizer.step()
   
        # for debugging 
        if np.isnan(np.asarray(model.module.dc2_conv7.weight.max().detach().cpu())):
            pdb.set_trace()
            pass

        vis = {}
        vis['output2'] = output[0].detach().cpu().numpy()
        vis['output3'] = output[1].detach().cpu().numpy()
        vis['output4'] = output[2].detach().cpu().numpy()
        vis['output5'] = output[3].detach().cpu().numpy()
        vis['output6'] = output[4].detach().cpu().numpy()
        if 'expansion' in args.stage:
            vis['mid'] = output[6][0].detach().cpu().numpy()
            vis['exp'] = output[7][0].detach().cpu().numpy()
        elif 'seg' in args.stage:
            vis['fg'] = output[6][0].detach().cpu().numpy()
            vis['fg_gt'] = output[7][0].detach().cpu().numpy()
        vis['gt'] = flowl0[:,:,:,:].detach().cpu().numpy()
        if mask.sum():
            vis['AEPE'] = realEPE(output[0].detach(), flowl0.permute(0,3,1,2).detach(),mask,sparse=False)
        vis['mask'] = mask
        vis['grad_norm'] = grad_norm
        return loss.data,vis

# get global counts                
with open('%s/iter_counts-%d.txt'%(args.itersave, int(args.logname.split('-')[-1])), 'w') as f:
    f.write('%d'%total_iters)

def main():
    sampler = torch.utils.data.distributed.DistributedSampler(
        data_inuse,
        num_replicas=args.nproc,
        rank=args.local_rank,
    )  
    TrainImgLoader = torch.utils.data.DataLoader(
         data_inuse, 
         batch_size= batch_size, num_workers=int(worker_mul*batch_size), drop_last=True, worker_init_fn=_init_fn, pin_memory=True,sampler=sampler)
    start_full_time = time.time()
    global total_iters

    # training loop
    for batch_idx, databatch in enumerate(TrainImgLoader):
        if 'expansion' in args.stage or 'seg' in args.stage:
            imgL_crop, imgR_crop, flowl0,imgAux,intr, imgoL, imgoR, occp, RT01  = databatch
            intr = [t.float() for t in intr]
        else:
            imgL_crop, imgR_crop, flowl0 = databatch
            imgAux,intr, imgoL, imgoR, occp, RT01 = None,None,None,None,None,None
        if total_iters < 1000 and not ('expansion' in args.stage or 'seg' in args.stage):
            # subtract mean
            mean_L.append( np.asarray(imgL_crop.mean(0).mean(1).mean(1)) )
            mean_R.append( np.asarray(imgR_crop.mean(0).mean(1).mean(1)) )
        imgL_crop -= torch.from_numpy(np.asarray(mean_L).mean(0)[np.newaxis,:,np.newaxis, np.newaxis]).float()
        imgR_crop -= torch.from_numpy(np.asarray(mean_R).mean(0)[np.newaxis,:,np.newaxis, np.newaxis]).float()

        start_time = time.time() 
        loss,vis = train(imgL_crop,imgR_crop, flowl0, imgAux,intr, imgoL, imgoR, occp, RT01)

        if args.local_rank==0:        
            print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
            if total_iters %10 == 0:
                log.add_scalar('train/loss_batch',loss, total_iters)
                log.add_scalar('train/aepe_batch',vis['AEPE'], total_iters)
                log.add_scalar('train/grad_norm',vis['grad_norm'], total_iters)
            if total_iters %100 == 0:
                #torch.cuda.empty_cache()
                add_image(log,'train/left', imgL_crop[0:1],total_iters)
                add_image(log,'train/right',imgR_crop[0:1],total_iters)
                if len(np.asarray(vis['gt']))>0:
                    log.add_histogram('train/gt_hist',np.asarray(vis['gt']).reshape(-1,3)[np.asarray(vis['gt'])[:,:,:,-1].flatten().astype(bool)][:,:2], total_iters)
                gu = vis['gt'][0,:,:,0]; gv = vis['gt'][0,:,:,1]
                gu = gu*np.asarray(vis['mask'][0].float().cpu());  gv = gv*np.asarray(vis['mask'][0].float().cpu())
                mask = vis['mask'][0].float().cpu()
                add_image(log,'train/gt0', flow_to_image(np.concatenate((gu[:,:,np.newaxis],gv[:,:,np.newaxis],mask[:,:,np.newaxis]),-1))[np.newaxis],total_iters)
                add_image(log,'train/output2',flow_to_image(vis['output2'][0].transpose((1,2,0)))[np.newaxis],total_iters)
                add_image(log,'train/output3',flow_to_image(vis['output3'][0].transpose((1,2,0)))[np.newaxis],total_iters)
                add_image(log,'train/output4',flow_to_image(vis['output4'][0].transpose((1,2,0)))[np.newaxis],total_iters)
                add_image(log,'train/output5',flow_to_image(vis['output5'][0].transpose((1,2,0)))[np.newaxis],total_iters)
                add_image(log,'train/output6',flow_to_image(vis['output6'][0].transpose((1,2,0)))[np.newaxis],total_iters)
                if 'expansion' in args.stage:
                    add_image(log,'train/mid_gt',(1+imgAux[:1,:,:,6]/imgAux[:1,:,:,0]).log() ,total_iters)
                    add_image(log,'train/mid',vis['mid'][np.newaxis],total_iters)
                    add_image(log,'train/exp',vis['exp'][np.newaxis],total_iters)
                if 'seg' in args.stage:
                    add_image(log,'train/fg_gt',vis['fg_gt'][np.newaxis],total_iters)
                    add_image(log,'train/fg_pred',vis['fg'][np.newaxis],total_iters)
        total_iters += 1
        # get global counts                
        with open('%s/iter_counts-%d.txt'%(args.itersave,int(args.logname.split('-')[-1])), 'w') as f:
            f.write('%d'%total_iters)
#        torch.cuda.empty_cache()

        if (total_iters + 1)%2000==0:
            if args.local_rank==0:
                #SAVE
                savefilename = args.savemodel+'/'+args.logname+'/finetune_'+str(total_iters)+'.pth'
                save_dict = model.state_dict()
                save_dict = collections.OrderedDict({k:v for k,v in save_dict.items() if ('reg_modules' not in k or 'conv1' in k) and ('grid' not in k) and ('flow_reg' not in k) and ('midas' not in k)  })
                torch.save({
                    'iters': total_iters,
                    'state_dict': save_dict,
                    'mean_L': mean_L,
                    'mean_R': mean_R,
                }, savefilename)
        
    print('full finetune time = %.2f HR' %((time.time() - start_full_time)/3600))


if __name__ == '__main__':
    main()
