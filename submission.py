from __future__ import print_function
import os
import sys
import cv2
import pdb
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import time
from utils.io import mkdir_p
from utils.util_flow import save_pfm, write_flow
from utils.flowlib import write_flo, point_vec
from dataloader.exploader import disparity_loader
from utils import dydepth as ddlib
cudnn.benchmark = False

parser = argparse.ArgumentParser(description='RigidMask')
parser.add_argument('--dataset', default='2015',
                    help='{2015, 2015val, sintelval, seq-XXX}')
parser.add_argument('--datapath', default='/ssd/kitti_scene/training/',
                    help='dataset path')
parser.add_argument('--loadmodel', default=None,
                    help='model path')
parser.add_argument('--outdir', default='output',
                    help='output dir')
parser.add_argument('--testres', type=float, default=1,
                    help='resolution')
parser.add_argument('--maxdisp', type=int ,default=256,
                    help='maxium disparity. Only affect the coarsest cost volume size')
parser.add_argument('--fac', type=float ,default=1,
                    help='controls the shape of search grid. Only affect the coarse cost volume size')
parser.add_argument('--disp_path', default='',
                    help='disparity input (only used for stereo)')
parser.add_argument('--mask_path', default='',
                    help='mask input')
parser.add_argument('--refine', dest='refine', action='store_true',
                    help='refine scene flow by rigid body motion')
parser.add_argument('--sensor', default='mono',
                    help='{mono} or stereo, will affect rigid motion parameterization')
args = parser.parse_args()


# dataloader
if args.dataset == '2015':
    from dataloader import kitti15list as DA
    maxw,maxh = [int(args.testres*1280), int(args.testres*384)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == '2015val':
    from dataloader import kitti15list_val as DA
    maxw,maxh = [int(args.testres*1280), int(args.testres*384)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == '2015vallidar':
    from dataloader import kitti15list_val_lidar as DA
    maxw,maxh = [int(args.testres*1280), int(args.testres*384)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == '2015test':
    from dataloader import kitti15list as DA
    maxw,maxh = [int(args.testres*1280), int(args.testres*384)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif 'seq' in args.dataset:
    from dataloader import seqlist as DA
    maxw,maxh = [int(args.testres*1280), int(args.testres*384)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == 'sinteltemple':
    from dataloader import sintel_temple as DA
    maxw,maxh = [int(args.testres*1024), int(args.testres*448)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == 'sinteltest':
    from dataloader import sintellist as DA
    maxw,maxh = [int(args.testres*1024), int(args.testres*448)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == 'sintel':
    from dataloader import  sintel_mrflow_val as DA
    #from dataloader import sintellist as DA
    maxw,maxh = [int(args.testres*1024), int(args.testres*448)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == 'sinteldepth':
    from dataloader import  sintel_rtn_val as DA
    maxw,maxh = [int(args.testres*1024), int(args.testres*448)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == 'sintelval':
    from dataloader import sintellist_val as DA
    maxw,maxh = [int(args.testres*1024), int(args.testres*448)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == 'mosegsintel':
    from dataloader import moseg_sintellist_val as DA
    maxw,maxh = [int(args.testres*1024), int(args.testres*448)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == 'mb':
    from dataloader import mblist as DA
    maxw,maxh = [int(args.testres*1024), int(args.testres*448)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == 'hd1k':
    from dataloader import hd1klist_val as DA
    maxw,maxh = [int(args.testres*1024), int(args.testres*448)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == 'viper':
    from dataloader import viperlist_val as DA
    maxw,maxh = [int(args.testres*1024), int(args.testres*448)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == 'viper_test':
    from dataloader import viperlist_test as DA
    maxw,maxh = [int(args.testres*1024), int(args.testres*448)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == 'tum':
    from dataloader import tumlist as DA
    maxw,maxh = [int(args.testres*1024), int(args.testres*448)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  

max_h = int(maxh // 64 * 64)
max_w = int(maxw // 64 * 64)
if max_h < maxh: max_h += 64
if max_w < maxw: max_w += 64
maxh = max_h
maxw = max_w


mean_L = [[0.33,0.33,0.33]]
mean_R = [[0.33,0.33,0.33]]

# construct model, VCN-expansion
from models.VCNplus import VCN
model = VCN([1, maxw, maxh], md=[int(4*(args.maxdisp/256)),4,4,4,4], fac=args.fac,exp_unc=not ('kitti' in args.loadmodel))
model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel,map_location='cpu')
    mean_L=pretrained_dict['mean_L']
    mean_R=pretrained_dict['mean_R']
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items()}
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)
else:
    print('dry run')
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

# load intrinsics calib
if 'seq' in args.dataset: 
    calib_path = '%s-calib.txt'%(args.datapath.rsplit('/',1)[0])
    if os.path.exists(calib_path):
        seqcalib = np.loadtxt(calib_path)
    else:
        exit()

mkdir_p('%s/%s/'% (args.outdir, args.dataset))
def main():
    model.eval()
    ttime_all = []
    for inx in range(len(test_left_img)):
        idxname = test_left_img[inx].split('/')[-1].split('.')[0]
        print(test_left_img[inx])
        imgL_o = cv2.imread(test_left_img[inx])[:,:,::-1]
        imgR_o = cv2.imread(test_right_img[inx])[:,:,::-1]

        # for gray input images
        if len(imgL_o.shape) == 2:
            imgL_o = np.tile(imgL_o[:,:,np.newaxis],(1,1,3))
            imgR_o = np.tile(imgR_o[:,:,np.newaxis],(1,1,3))

        # resize
        maxh = imgL_o.shape[0]*args.testres
        maxw = imgL_o.shape[1]*args.testres
        max_h = int(maxh // 64 * 64)
        max_w = int(maxw // 64 * 64)
        if max_h < maxh: max_h += 64
        if max_w < maxw: max_w += 64

        input_size = imgL_o.shape
        imgL = cv2.resize(imgL_o,(max_w, max_h))
        imgR = cv2.resize(imgR_o,(max_w, max_h))
        imgL_noaug = torch.Tensor(imgL/255.)[np.newaxis].float().cuda()

        # flip channel, subtract mean
        imgL = imgL[:,:,::-1].copy() / 255. - np.asarray(mean_L).mean(0)[np.newaxis,np.newaxis,:]
        imgR = imgR[:,:,::-1].copy() / 255. - np.asarray(mean_R).mean(0)[np.newaxis,np.newaxis,:]
        imgL = np.transpose(imgL, [2,0,1])[np.newaxis]
        imgR = np.transpose(imgR, [2,0,1])[np.newaxis]

        # modify module according to inputs
        from models.VCNplus import WarpModule, flow_reg
        for i in range(len(model.module.reg_modules)):
            model.module.reg_modules[i] = flow_reg([1,max_w//(2**(6-i)), max_h//(2**(6-i))], 
                            ent=getattr(model.module, 'flow_reg%d'%2**(6-i)).ent,\
                            maxdisp=getattr(model.module, 'flow_reg%d'%2**(6-i)).md,\
                            fac=getattr(model.module, 'flow_reg%d'%2**(6-i)).fac).cuda()
        for i in range(len(model.module.warp_modules)):
            model.module.warp_modules[i] = WarpModule([1,max_w//(2**(6-i)), max_h//(2**(6-i))]).cuda()

        # get intrinsics
        if '2015' in args.dataset:
            from utils.util_flow import load_calib_cam_to_cam
            ints = load_calib_cam_to_cam(test_left_img[inx].replace('image_2','calib_cam_to_cam')[:-7]+'.txt')
            K0 = ints['K_cam2']
            K1 = K0
            fl = K0[0,0]
            cx = K0[0,2]
            cy = K0[1,2]
            bl = ints['b20']-ints['b30']
            fl_next = fl
            intr_list = [torch.Tensor(inxx).cuda() for inxx in [[fl],[cx],[cy],[bl],[1],[0],[0],[1],[0],[0]]]
        elif 'sintel' in args.dataset and not 'test' in test_left_img[inx]:
            from utils.sintel_io import cam_read
            passname = test_left_img[inx].split('/')[-1].split('_')[-4]
            seqname1 = test_left_img[inx].split('/')[-1].split('_')[-3]
            seqname2 = test_left_img[inx].split('/')[-1].split('_')[-2]
            framename = int(test_left_img[inx].split('/')[-1].split('_')[-1].split('.')[0])
            #TODO add second camera
            K0,_ = cam_read('/data/gengshay/tf_depth/sintel-data/training/camdata_left/%s_%s/frame_%04d.cam'%(seqname1, seqname2, framename+1))
            K1,_ = cam_read('/data/gengshay/tf_depth/sintel-data/training/camdata_left/%s_%s/frame_%04d.cam'%(seqname1, seqname2, framename+2))
            fl = K0[0,0]
            cx = K0[0,2]
            cy = K0[1,2]
            fl_next = K1[0,0]
            bl = 0.1
            intr_list = [torch.Tensor(inxx).cuda() for inxx in [[fl],[cx],[cy],[bl],[1],[0],[0],[1],[0],[0]]]
        elif 'seq' in args.dataset:
            fl,cx,cy = seqcalib[inx]
            bl = 1
            fl_next = fl
            K0 = np.eye(3)
            K0[0,0] = fl
            K0[1,1] = fl
            K0[0,2] = cx
            K0[1,2] = cy
            K1 = K0
            intr_list = [torch.Tensor(inxx).cuda() for inxx in [[fl],[cx],[cy],[bl],[1],[0],[0],[1],[0],[0]]]
        else:
            print('NOT using given intrinsics')
            fl = min(input_size[0], input_size[1]) *2
            fl_next = fl
            cx = input_size[1]/2.
            cy = input_size[0]/2.
            bl = 1
            K0 = np.eye(3)
            K0[0,0] = fl
            K0[1,1] = fl
            K0[0,2] = cx
            K0[1,2] = cy
            K1 = K0
            intr_list = [torch.Tensor(inxx).cuda() for inxx in [[fl],[cx],[cy],[bl],[1],[0],[0],[1],[0],[0]]]
        intr_list.append(torch.Tensor([input_size[1] / max_w]).cuda()) # delta fx
        intr_list.append(torch.Tensor([input_size[0] / max_h]).cuda()) # delta fy
        intr_list.append(torch.Tensor([fl_next]).cuda())
        
        disc_aux = [None,None,None,intr_list,imgL_noaug,None]
        
        if args.disp_path=='': disp_input=None
        else:
            try:
                disp_input = disparity_loader('%s/%s_disp.pfm'%(args.disp_path,idxname)) 
            except:
                disp_input = disparity_loader('%s/%s.png'%(args.disp_path,idxname))
            disp_input = torch.Tensor(disp_input.copy())[np.newaxis,np.newaxis].cuda()

        # forward
        imgL = Variable(torch.FloatTensor(imgL).cuda())
        imgR = Variable(torch.FloatTensor(imgR).cuda())
        with torch.no_grad():
            imgLR = torch.cat([imgL,imgR],0)
            model.eval()
            torch.cuda.synchronize()
            start_time = time.time()
            rts = model(imgLR, disc_aux, disp_input)
            torch.cuda.synchronize()
            ttime = (time.time() - start_time); print('time = %.2f' % (ttime*1000) )
            ttime_all.append(ttime)
            flow, occ, logmid, logexp, fgmask, heatmap, polarmask, disp = rts
            bbox = polarmask['bbox']
            polarmask = polarmask['mask']
            polarcontour = polarmask[:polarmask.shape[0]//2]        
            polarmask = polarmask[polarmask.shape[0]//2:]

        # upsampling
        occ = cv2.resize(occ.data.cpu().numpy(),  (input_size[1],input_size[0]),interpolation=cv2.INTER_LINEAR)
        logexp = cv2.resize(logexp.cpu().numpy(), (input_size[1],input_size[0]),interpolation=cv2.INTER_LINEAR)
        logmid = cv2.resize(logmid.cpu().numpy(), (input_size[1],input_size[0]),interpolation=cv2.INTER_LINEAR)
        fgmask = cv2.resize(fgmask.cpu().numpy(), (input_size[1],input_size[0]),interpolation=cv2.INTER_LINEAR)
        heatmap= cv2.resize(heatmap.cpu().numpy(), (input_size[1],input_size[0]),interpolation=cv2.INTER_LINEAR)
        polarcontour= cv2.resize(polarcontour, (input_size[1],input_size[0]),interpolation=cv2.INTER_NEAREST)
        polarmask= cv2.resize(polarmask, (input_size[1],input_size[0]),interpolation=cv2.INTER_NEAREST).astype(int)
        polarmask[np.logical_and(fgmask>0,polarmask==0)]=-1
        if args.disp_path=='':
            disp= cv2.resize(disp.cpu().numpy(), (input_size[1],input_size[0]),interpolation=cv2.INTER_LINEAR)
        else:
            disp = np.asarray(disp_input.cpu())[0,0]
        flow = torch.squeeze(flow).data.cpu().numpy()
        flow = np.concatenate( [cv2.resize(flow[0],(input_size[1],input_size[0]))[:,:,np.newaxis],
                                cv2.resize(flow[1],(input_size[1],input_size[0]))[:,:,np.newaxis]],-1)
        flow[:,:,0] *= imgL_o.shape[1] / max_w
        flow[:,:,1] *= imgL_o.shape[0] / max_h
        flow = np.concatenate( (flow, np.ones([flow.shape[0],flow.shape[1],1])),-1)
        bbox[:,0] *= imgL_o.shape[1] / max_w
        bbox[:,2] *= imgL_o.shape[1] / max_w
        bbox[:,1] *= imgL_o.shape[0] / max_h
        bbox[:,3] *= imgL_o.shape[0] / max_h
        
        # draw instance center and motion in 2D
        ins_center_vis = np.zeros(flow.shape[:2])
        for k in range(bbox.shape[0]):
            from utils.detlib import draw_umich_gaussian
            draw_umich_gaussian(ins_center_vis, bbox[k,:4].reshape(2,2).mean(0), 15)
        ins_center_vis = 256*np.stack([ins_center_vis, np.zeros(ins_center_vis.shape), np.zeros(ins_center_vis.shape)],-1)
        if args.refine:
            ## depth and scene flow estimation
            # save initial disp and flow
            init_disp = disp.copy()
            init_flow = flow.copy()
            init_logmid = logmid.copy()

            if args.mask_path == '':
                mask_input = polarmask
            else:
                mask_input = cv2.imread('%s/%s.png'%(args.mask_path,idxname),0)
                if mask_input is None:
                    mask_input = cv2.imread('%s/%s.png'%(args.mask_path,idxname.split('_')[0]),0)
                
            bgmask = (mask_input == 0) 
            scene_type, T01_c, R01,RTs = ddlib.rb_fitting(bgmask,mask_input,disp,flow,occ,K0,K1,bl,parallax_th=4,mono=(args.sensor=='mono'), sintel='Sintel' in idxname)
            print('camera trans: '); print(T01_c)
            disp,flow,disp1 = ddlib.mod_flow(bgmask,mask_input,disp,disp/np.exp(logmid),flow,occ,bl,K0,K1,scene_type, T01_c,R01, RTs, fgmask,mono=(args.sensor=='mono'), sintel='Sintel' in idxname)
            logmid = np.clip(np.log(disp / disp1),-1,1)

            # draw ego vehicle
            ct = [4*input_size[0]//5,input_size[1]//2][::-1] 
            cv2.circle(ins_center_vis, tuple(ct), radius=10,color=(0,255,255),thickness=10)
            obj_3d = K0[0,0]*bl/np.median(disp[bgmask]) * np.linalg.inv(K0).dot(np.hstack([ct,np.ones(1)]))
            obj_3d2 = obj_3d + (-R01.T.dot(T01_c))
            ed = K0.dot(obj_3d2)
            ed = (ed[:2]/ed[-1]).astype(int)
            if args.sensor=='mono':
                direct = (ed - ct)
                direct = 50*direct/(1e-9+np.linalg.norm(direct))
            else:
                direct = (ed - ct)
            ed = (ct+direct).astype(int)
            if np.linalg.norm(direct)>1:
                ins_center_vis = cv2.arrowedLine(ins_center_vis, tuple(ct), tuple(ed), (0,255,255),6,tipLength=float(30./np.linalg.norm(direct)))

            # draw each object
            for k in range(mask_input.max()):
                try:
                    obj_mask = mask_input==k+1
                    if obj_mask.sum()==0:continue
                    ct = np.asarray(np.nonzero(obj_mask)).mean(1).astype(int)[::-1] # Nx2
                    cv2.circle(ins_center_vis, tuple(ct), radius=5,color=(255,0,0),thickness=5)
                    if RTs[k] is not None:
                        #ins_center_vis[mask_input==k+1] = imgL_o[mask_input==k+1]
                        obj_3d = K0[0,0]*bl/np.median(disp[mask_input==k+1]) * np.linalg.inv(K0).dot(np.hstack([ct,np.ones(1)]))
                        obj_3d2 = obj_3d + (-RTs[k][0].T.dot(RTs[k][1]) )
                        ed = K0.dot(obj_3d2)
                        ed = (ed[:2]/ed[-1]).astype(int)
                        if args.sensor=='mono':
                            direct = (ed - ct)
                            direct = 50*direct/(np.linalg.norm(direct)+1e-9)
                        else:
                            direct = (ed - ct)
                        ed = (ct+direct).astype(int)
                        if np.linalg.norm(direct)>1:
                            ins_center_vis = cv2.arrowedLine(ins_center_vis, tuple(ct), tuple(ed), (255,0,0),3,tipLength=float(30./np.linalg.norm(direct)))  
                except:pdb.set_trace()
        cv2.imwrite('%s/%s/mvis-%s.jpg'% (args.outdir, args.dataset,idxname), ins_center_vis[:,:,::-1])

        # save predictions
        with open('%s/%s/flo-%s.pfm'% (args.outdir, args.dataset,idxname),'w') as f:
            save_pfm(f,flow[::-1].astype(np.float32))
        # flow vis: visualization of 2d flow vectors in the rgb space.
        flowvis = point_vec(imgL_o, flow)
        cv2.imwrite('%s/%s/visflo-%s.jpg'% (args.outdir, args.dataset,idxname),flowvis)
        imwarped = ddlib.warp_flow(imgR_o, flow[:,:,:2])
        cv2.imwrite('%s/%s/warp-%s.jpg'% (args.outdir, args.dataset,idxname),imwarped[:,:,::-1])
        cv2.imwrite('%s/%s/warpt-%s.jpg'% (args.outdir, args.dataset,idxname),imgL_o[:,:,::-1])
        cv2.imwrite('%s/%s/warps-%s.jpg'% (args.outdir, args.dataset,idxname),imgR_o[:,:,::-1])
        with open('%s/%s/occ-%s.pfm'% (args.outdir, args.dataset,idxname),'w') as f:
            save_pfm(f,occ[::-1].astype(np.float32))
        with open('%s/%s/exp-%s.pfm'% (args.outdir, args.dataset,idxname),'w') as f:
            save_pfm(f,logexp[::-1].astype(np.float32))
        with open('%s/%s/mid-%s.pfm'% (args.outdir, args.dataset,idxname),'w') as f:
            save_pfm(f,logmid[::-1].astype(np.float32))
        with open('%s/%s/fg-%s.pfm'% (args.outdir, args.dataset,idxname),'w') as f:
            save_pfm(f,fgmask[::-1].astype(np.float32))
        with open('%s/%s/hm-%s.pfm'% (args.outdir, args.dataset,idxname),'w') as f:
            save_pfm(f,heatmap[::-1].astype(np.float32))
        with open('%s/%s/pm-%s.pfm'% (args.outdir, args.dataset,idxname),'w') as f:
            save_pfm(f,polarmask[::-1].astype(np.float32))
        ddlib.write_calib(K0,bl,polarmask.shape, K0[0,0]*bl / (np.median(disp)/5),
                    '%s/%s/calib-%s.txt'% (args.outdir, args.dataset,idxname))
        
        # submit to KITTI benchmark
        if 'test' in args.dataset:
            outdir = 'benchmark_output'
            # kitti scene flow
            import skimage.io
            skimage.io.imsave('%s/disp_0/%s.png'% (outdir,idxname),(disp*256).astype('uint16'))
            skimage.io.imsave('%s/disp_1/%s.png'% (outdir,idxname),(disp1*256).astype('uint16'))
            flow[:,:,2]=1.
            write_flow(       '%s/flow/%s.png'% (outdir,idxname.split('.')[0]),flow)

        # save visualizations
        with open('%s/%s/disp-%s.pfm'% (args.outdir, args.dataset,idxname),'w') as f:
            save_pfm(f,disp[::-1].astype(np.float32))

        try:
            # point clouds
            from utils.fusion import pcwrite
            hp2d0 = np.concatenate( [np.tile(np.arange(0, input_size[1]).reshape(1,-1),(input_size[0],1)).astype(float)[None],  # 1,2,H,W
                                     np.tile(np.arange(0, input_size[0]).reshape(-1,1),(1,input_size[1])).astype(float)[None],
                                     np.ones(input_size[:2])[None]], 0).reshape(3,-1)
            hp2d1 = hp2d0.copy()
            hp2d1[:2] += np.transpose(flow,[2,0,1])[:2].reshape(2,-1)
            p3d0 = (K0[0,0]*bl/disp.flatten()) * np.linalg.inv(K0).dot(hp2d0)
            p3d1 = (K0[0,0]*bl/disp1.flatten()) * np.linalg.inv(K1).dot(hp2d1)
            def write_pcs(points3d, imgL_o,mask_input,path):
                # remove some points
                points3d = points3d.T.reshape(input_size[:2]+(3,))
                points3d[points3d[:,:,-1]>np.median(points3d[:,:,-1])*5]=0
                #points3d[:2*input_size[0]//5] = 0. # KITTI
                points3d = np.concatenate([points3d, imgL_o],-1)
                validid = np.linalg.norm(points3d[:,:,:3],2,-1) >0
                bgidx = np.logical_and(validid, mask_input==0)
                fgidx = np.logical_and(validid, mask_input>0)
                pcwrite(path.replace('/pc', '/fgpc'), points3d[fgidx])
                pcwrite(path.replace('/pc', '/bgpc'), points3d[bgidx])
                pcwrite(path, points3d[validid])
            if inx==0:
                write_pcs(p3d0,imgL_o,mask_input,path='%s/%s/pc0-%s.ply'% (args.outdir, args.dataset,idxname))
                write_pcs(p3d1,imgL_o,mask_input,path='%s/%s/pc1-%s.ply'% (args.outdir, args.dataset,idxname))
        except:pass
        torch.cuda.empty_cache()
    print(np.mean(ttime_all))
                
            

if __name__ == '__main__':
    main()

