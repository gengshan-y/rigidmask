import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import pdb
import glob

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

  left_fold  = 'image_2/'
  flow_train = sorted(glob.glob('%s/flow_occ/*'%filepath))
  l0_train = [img.replace('/flow_occ/', '/image_2/') for img in flow_train]
  l1_train = ['%s_%06d.png'%(img.rsplit('_', 1)[0], 1+int(img.split('_')[-1].split('.png')[0])) for img in l0_train]

  return l0_train, l1_train, flow_train
