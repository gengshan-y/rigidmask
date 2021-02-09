import numpy as np
import trimesh
import torch
import cv2
import kornia
import pdb
import glob
import imageio
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--inpath', default='weights/rigidmask-sf/seq-kitti/pc0-0000001110.ply',
                    help='data path')
parser.add_argument('--outdir', default='./',
                    help='data path')
args = parser.parse_args()

## io
img_size = 1024
nframes=100

proj_mat = torch.eye(4)[np.newaxis].cuda()

# pytorch3d 
from pytorch3d.renderer.points import (
    AlphaCompositor)
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
 PointLights, 
 PointsRasterizationSettings, 
 PointsRenderer, 
 PointsRasterizer,  
 )
from pytorch3d.renderer.cameras import OrthographicCameras
device = torch.device("cuda:0") 

pclist = sorted(glob.glob('%s*'%(args.inpath)))
print(pclist[0])
mesh = trimesh.load(pclist[0])
mesh.vertices -= mesh.vertices.mean(0)[None]
mesh.vertices[:,:2] *= -1
overts = torch.Tensor( mesh.vertices).cuda()

frames = []
for i in range(0,nframes):
    print(i)
    try:
        # dynamic
        mesh = trimesh.load(pclist[i])
        mesh.vertices[:,:2] *= -1
        overts = torch.Tensor( mesh.vertices).cuda()
    except:pass
    # extract camera in/ex
    verts = overts.clone()
    #rotmat = cv2.Rodrigues(np.asarray([6.28*i/nframes,0.,0.]))[0]  # x-axis
    #rotmat = cv2.Rodrigues(np.asarray([-0.05*6.28, 0.10*6.28*(-0.5+i/nframes),0.]))[0]  # y-axis
    rotmat = cv2.Rodrigues(np.asarray([-0.02*6.28, 0.25*6.28*(-0.5+i/nframes),0.]))[0]  # y-axis
    #rotmat = cv2.Rodrigues(np.asarray([-0.01*6.28, 0.10*6.28*(-0.5+i/nframes),0.]))[0]  # y-axis temple
    #rotmat = cv2.Rodrigues(np.asarray([ 0.05*6.28, 0.10*6.28*(-0.5+i/nframes),0.]))[0]  # y-axis temple
    quat = kornia.rotation_matrix_to_quaternion(torch.Tensor(rotmat).cuda())
    proj_cam = torch.zeros(1,7).cuda()
    depth = torch.zeros(1,1).cuda()
    proj_cam[:,0]=5   # focal=10 
    proj_cam[:,1] = 0. # x translation = 0
    proj_cam[:,2] = 0. # y translation = 0
    proj_cam[:,3]=quat[3]
    proj_cam[:,4:]=quat[:3]
    #depth[:,0] = 0.05   # for temple 
    depth[:,0] = 1   # z translation (depth) =10 for spot
    #depth[:,0] = 0.5   # z translation (depth) =10 for spot
    #depth[:,0] = 200   # for kitti

    # obj-cam transform 
    Rmat = kornia.quaternion_to_rotation_matrix(torch.cat((-proj_cam[:,4:],proj_cam[:,3:4]),1))
    Tmat = torch.cat([proj_cam[:,1:3],depth],1)
    verts = verts.matmul(Rmat) + Tmat[:,np.newaxis,:]  # obj to cam transform
    verts = torch.cat([verts,torch.ones_like(verts[:, :, 0:1])], dim=-1)
    
    # pespective projection: x=fX/Z assuming px=py=0, normalization of Z
    verts[:,:,1] = verts[:, :, 1].clone()*proj_cam[:,:1]/ verts[:,:,2].clone()
    verts[:,:,0] = verts[:, :, 0].clone()*proj_cam[:,:1]/ verts[:,:,2].clone()
    verts[:,:,2] = ( (verts[:,:,2]-verts[:,:,2].min())/(verts[:,:,2].max()-verts[:,:,2].min())-0.5).detach()
    verts[:,:,2] += 10

    features = torch.ones_like(verts)
    point_cloud = Pointclouds(points=verts[:,:,:3], features=torch.Tensor(mesh.visual.vertex_colors[None]).cuda())

    cameras = OrthographicCameras(device=device)
    raster_settings = PointsRasterizationSettings(
        image_size=img_size, 
        radius = 0.005,
        points_per_pixel = 10
    )
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=(33,33,33))
    )
    img_pred = renderer(point_cloud)
    frames.append(img_pred[0,:,:,:3].cpu())
    #cv2.imwrite('%s/points%04d.png'%(args.outdir,i), np.asarray(img_pred[0,:,:,:3].cpu())[:,:,::-1])    
imageio.mimsave('./output-depth.gif', frames, duration=5./len(frames))
