import numpy as np
import pdb
from matplotlib import pyplot as plt 
import sys
import scipy.stats as ss

with open(sys.argv[1],'r') as f:
    lines = f.readlines()
niter = []
em1 = []
em2 = []
eh1 = []
eh2 = []
ek1 = []
ek2 = []
es1 = []
es2 = []
ev1 = []
ev2 = []
for i,l in enumerate(lines):
    if l[:4] != 'flow' and len(lines)>(i+4):
        niter.append(lines[i])
        em1.append(float(lines[i+1].split('%/')[0].split(':')[-1]))
        em2.append(float(lines[i+1].split('%/')[1].split('px')[0]))
        eh1.append(float(lines[i+2].split('%/')[0].split(':')[-1]))
        eh2.append(float(lines[i+2].split('%/')[1].split('px')[0]))
        ek1.append(float(lines[i+3].split('%/')[0].split(':')[-1]))
        ek2.append(float(lines[i+3].split('%/')[1].split('px')[0]))
        ev1.append(float(lines[i+4].split('%/')[0].split(':')[-1]))
        ev2.append(float(lines[i+4].split('%/')[1].split('px')[0]))
        es1.append(float(lines[i+5].split('%/')[0].split(':')[-1]))
        es2.append(float(lines[i+5].split('%/')[1].split('px')[0]))

rem1 = ss.rankdata(em1)
reh1 = ss.rankdata(eh1)
rek1 = ss.rankdata(ek1)
res1 = ss.rankdata(es1)
rev1 = ss.rankdata(ev1)
reavg1 = np.stack([rem1, reh1, res1, rek1, rev1]).mean(0)
rem2 = ss.rankdata(em2)
reh2 = ss.rankdata(eh2)
rek2 = ss.rankdata(ek2)
res2 = ss.rankdata(es2)
rev2 = ss.rankdata(ev2)
reavg2 = np.stack([rem2, reh2, res2, rek2,rev2]).mean(0)
ravg_all = np.stack([reavg1, reavg2]).mean(0)

plt.figure(figsize=(12,4))
plt.plot(rem1,  '.-',label='Rank of Middlebury: Fl-err')
plt.plot(reh1,  '.-',label='Rank of HD1K Fl-err')
plt.plot(rek1,  '.-',label='Rank of KITTI Fl-err')
plt.plot(res1,  '.-',label='Rank of Sintel Fl-err')
plt.plot(rev1,  '.-',label='Rank of VIPER Fl-err')
plt.plot(reavg1,'o-',label='Rank of avg Fl-err')
plt.legend()
plt.savefig('rank1.png')

plt.figure(figsize=(12,4))
plt.plot(rem2,  '.-',label='Rank of Middlebury: EPE ')
plt.plot(reh2,  '.-',label='Rank of HD1K        EPE ')
plt.plot(rek2,  '.-',label='Rank of KITTI       EPE ')
plt.plot(res2,  '.-',label='Rank of Sintel      EPE ')
plt.plot(rev2,  '.-',label='Rank of VIPER      EPE ')
plt.plot(reavg2,'o-',label='Rank of avg         EPE')
plt.legend()
plt.savefig('rank2.png')

plt.figure(figsize=(12,4))
plt.plot(ravg_all,'*-',label='Rank of avg       all')
plt.legend()
plt.savefig('rank3.png')
miter = np.argsort(ravg_all)[0]
print('best iter:%s\t%.1f%%/%.3fpx\t%.1f%%/%.3fpx\t%.1f%%/%.3fpx\t%.1f%%/%.3fpx\t%.1f%%/%.3fpx' %(niter[miter],em1[miter],em2[miter],eh1[miter],eh2[miter],ek1[miter],ek2[miter],es1[miter],es2[miter],ev1[miter],ev2[miter] ))
