import os
import sys
sys.path.insert(0,os.getcwd())
import cv2
import numpy as np
import glob
import argparse
from utils.util_flow import readPFM

parser = argparse.ArgumentParser(description='')
parser.add_argument('--datapath', default='',
                    help='dataset path')
parser.add_argument('--outpath', default='',
                    help='output path')
args = parser.parse_args()

for folder in glob.glob('%s/seq-*'%(args.datapath)):
    name = folder.split('seq-')[-1]
    dirpath = '%s/%s/'%(args.outpath, name)
    print(dirpath)
    if not os.path.isdir(dirpath):       os.mkdir(dirpath)
    for filename in glob.glob('%s/pm*.pfm'%folder):
        frameid = filename.split('pm-')[-1].split('.pfm')[0]
        print(filename)
        mask_pred = readPFM(filename)[0].astype(int)>0
        shape = mask_pred.shape
        mask_pred = np.stack([np.zeros(shape), np.zeros(shape), mask_pred.astype(int)*128],-1)
        cv2.imwrite('%s/%s.png'%(dirpath, frameid), mask_pred)
