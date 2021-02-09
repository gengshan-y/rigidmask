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
  train = [img for img in os.listdir(filepath+left_fold) if img.find('Sintel') > -1]

  l0_train  = [filepath+left_fold+img for img in train]
  # check if the second frame exists
  l0_train  = [img for img in l0_train if '%s_%s.png'%(img.rsplit('_',1)[0],'%02d'%(1+int(img.split('.')[0].split('_')[-1])) ) in l0_train ]
  
  new_l0_train = []
  for fp in l0_train:
    passname = fp.split('/')[-1].split('_')[-4]
    seqname1 = fp.split('/')[-1].split('_')[-3]
    seqname2 = fp.split('/')[-1].split('_')[-2]
    framename = int(fp.split('/')[-1].split('_')[-1].split('.')[0])
    length = len(glob.glob('%s/%s/*%s_%s_%s*'%(filepath, left_fold,passname,seqname1,seqname2)) )
    if length - framename > 30:  # remove last 30 frames (which is used by MR-flow for training)
        new_l0_train.append(fp)
  l0_train = new_l0_train

  l1_train = ['%s_%s.png'%(img.rsplit('_',1)[0],'%02d'%(1+int(img.split('.')[0].split('_')[-1])) ) for img in l0_train]
  flow_train = [img.replace('image_2','flow_occ') for img in l0_train]


  return l0_train, l1_train, flow_train
