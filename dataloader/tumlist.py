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

  with open('%s/test_data/test_tum_hdf5_list.txt'%filepath, 'r') as f:
    list = f.readlines()
  train = ['%s/%s'%(filepath, img.replace('/tum_hdf5/', '/tum_img/').replace('.jpg.h5', '.jpg').strip()) for img in list]

  l0_train  = train[:-1]
  l1_train = train[1:]


  return l0_train, l1_train, l0_train
