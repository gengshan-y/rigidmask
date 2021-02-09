import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import glob

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):
    l0_train = []
    l1_train = []
    flow_train = []

    for seq in sorted(glob.glob('%s/*'%filepath)):
        flow_train += sorted(glob.glob('%s/flow0/*.pfm'%(seq)))
        l0_train += sorted(glob.glob('%s/im0/*.png'%(seq)))[:-1]
        l1_train += sorted(glob.glob('%s/im0/*.png'%(seq)))[1:]
        #print("%d %d %d"%(len(flow_train), len(l0_train), len(l1_train)))

        flow_train += sorted(glob.glob('%s/flow1/*.pfm'%(seq)))
        l0_train += sorted(glob.glob('%s/im0/*.png'%(seq)))[1:]
        l1_train += sorted(glob.glob('%s/im0/*.png'%(seq)))[:-1]

    return l0_train, l1_train, flow_train
