import numpy as np
import pdb
from matplotlib import pyplot as plt 
import sys
import scipy.stats as ss

with open(sys.argv[1],'r') as f:
    lines = f.readlines()
niter = []
ek1 = []
ek2 = []
es1 = []
for i,l in enumerate(lines):
    if l[:3] != 'IoU' and l[:3] != 'obj' and len(lines)>(i+3):
        niter.append(lines[i])
        ek1.append(100-float(lines[i+1].split(' ')[1]))
        ek2.append(100-float(lines[i+2].split(' ')[1]))
        es1.append(100-float(lines[i+3].split(' ')[1]))

rek1 = ss.rankdata(ek1)
rek2 = ss.rankdata(ek2)
res1 = ss.rankdata(es1)
reavg1 = np.stack([res1, rek1, rek2]).mean(0)

#plt.figure(figsize=(12,4))
#plt.plot(rem1,  '.-',label='Rank of Middlebury: Fl-err')
#plt.plot(reh1,  '.-',label='Rank of HD1K Fl-err')
#plt.plot(rek1,  '.-',label='Rank of KITTI Fl-err')
#plt.plot(res1,  '.-',label='Rank of Sintel Fl-err')
#plt.plot(rev1,  '.-',label='Rank of VIPER Fl-err')
#plt.plot(reavg1,'o-',label='Rank of avg Fl-err')
#plt.legend()
#plt.savefig('rank1.png')
#
#plt.figure(figsize=(12,4))
#plt.plot(rem2,  '.-',label='Rank of Middlebury: EPE ')
#plt.plot(reh2,  '.-',label='Rank of HD1K        EPE ')
#plt.plot(rek2,  '.-',label='Rank of KITTI       EPE ')
#plt.plot(res2,  '.-',label='Rank of Sintel      EPE ')
#plt.plot(rev2,  '.-',label='Rank of VIPER      EPE ')
#plt.plot(reavg2,'o-',label='Rank of avg         EPE')
#plt.legend()
#plt.savefig('rank2.png')
#
#plt.figure(figsize=(12,4))
#plt.plot(ravg_all,'*-',label='Rank of avg       all')
#plt.legend()
#plt.savefig('rank3.png')
miter = np.argsort(reavg1)[0]
if (reavg1 == reavg1[miter]).sum()>1: 
    miter = (np.where((reavg1 == reavg1[miter])))[0][-1]
print('best iter:%s\t%.1f (%%)/%.1f (%%)/%.1f (%%)' %(niter[miter],100-ek2[miter],100-ek1[miter], 100-es1[miter]))
