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
  train = glob.glob(filepath+left_fold+'/0*.jpg')
  train = sorted(train)

  l0_train = []
  l1_train = []
  flow_train = []
  for img in train:
    img1 = ('%s_%s.jpg'%(img.rsplit('_',1)[0],'%05d'%(1+int(img.split('.')[0].split('_')[-1])) ))
    flowp = img.replace('.jpg', '.png').replace('image_2','flow_occ')
    if (img1 in train and len(glob.glob(flowp))>0):
    #if (img1 in train and len(glob.glob(flowp))>0 and ('01000' not in img)):
      l0_train.append(img)   
      l1_train.append(img1)
      flow_train.append(flowp)

  return l0_train, l1_train, flow_train
