#!/usr/bin/env python

from __future__ import print_function
import sys
# fit_lightcurves also provides a `super` class which shadows the builtin
from fit_lightcurves import columns, super
import numpy as np

f = open('stretch.dat','w')

for name in sys.argv[1:]:
   data = columns(name)
   data2 = columns(name.replace("template","sigma"))

   bands = ['B','V','R','I']
   s = super('test')
   for i in range(4):
      s.data[bands[i]] = {}
      s.data[bands[i]]['MJD'] = data[0]
      s.data[bands[i]]['mag'] = data[i+1]
      s.data[bands[i]]['e_mag'] = data2[i+1]

   s.mask['B'] = np.greater(s.data['B']['MJD'], 0)
   s.z = 0
   s.get_restbands()
   s.s = 1
   gids = np
   s.fit(['B'], dm15=1.1, EBVhost=0, Tmax=0)
   
   print(name, s.s, file=f)

f.close()
