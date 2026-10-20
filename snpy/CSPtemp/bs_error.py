#!/usr/bin/env python

import sys,os,string
import numpy as np
from scipy.interpolate import bisplev, bisplrep
from myplotlib import PanelPlot
from matplotlib import pyplot
import pickle

tck_file0 = 'tck.pickle'
tck_file1 = 'bs_tck.pickle'
error_file = 'bs_error.pickle'

f = open(tck_file0)
all_tck0 = pickle.load(f)
f.close()

f = open(tck_file1)
all_tcks = pickle.load(f)
f.close()

dm15_low = 0.6
dm15_high = 2.0
t_low = -10.0
t_high = 70

dm15s = np.arange(31)/30.0*(dm15_high - dm15_low) + dm15_low
ts = np.arange(81)/80.0*(t_high - t_low) + t_low

tck = {}

bands = ['u','B','V','g','r','i','Y','J','H']
for j in range(len(bands)):
   band = bands[j]
   tck0 = all_tck0[band]
   tcks = all_tcks[band]
   t_mat = []
   dm15_mat = []
   dz_mat = []
   for k in range(len(dm15s)):
      t_mat.append(ts)
      dm15_mat.append(ts*0 + dm15s[k])
      dzs = []
      z0 = bisplev(ts, dm15s[k], tck0)[:,0]
      for i in range(len(tcks)):
         z1 = bisplev(ts, dm15s[k], tcks[i])[:,0]
         if np.any(np.absolute(z1 - z0) > 0.15):
            continue
         dzs.append(z1 - z0)
      dzs = np.array(dzs)
      rms = np.sqrt(np.mean(np.power(dzs, 2), axis=0))
      dz_mat.append(rms)
      mads = np.median(np.absolute(dzs), axis=0)
   tck[band] = bisplrep(np.ravel(np.array(t_mat)), np.ravel(np.array(dm15_mat)),
         np.ravel(np.array(dz_mat)), kx=1, ky=1, s=0)

f = open(error_file, 'w')
pickle.dump(tck, f)
f.close()

