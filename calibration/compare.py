'''Compare calibrations from calibrate.py with snpy's st_calibration2.dat.

Usage:
   python compare.py results/cal_2012_gen3.1_v{0..5}.json [results/cal_extended_gen3.1_v{0..5}.json ...]
'''
import sys, os, json
import numpy as np
from snpy.model import read_table, base

a0, ea0, b0, eb0, c0, ec0, Rv0, eRv0, sig0 = read_table(os.path.join(base, 'st_calibration2.dat'))
cals = [json.load(open(f)) for f in sys.argv[1:]]

print('R_V and sample size per variant (file -> new):')
for c in cals:
   v = c['variant']
   print('  %-9s v%d %-26s N=%3d  R_V %.2f(%.2f) -> %.2f(%.2f)' % (c['sample'], v, c['description'],
         c['N'], Rv0[v], eRv0[v], c['Rv'], c['eRv']))

for c in cals:
   if c['variant'] != 0:
      continue
   print('\n%s sample, variant 0: new - file (in units of the combined error)' % c['sample'])
   print('     %-22s %-22s %-22s %s' % ('a', 'b', 'c', 'sigma file/new'))
   for i, B in enumerate(c['bands']):
      row = []
      for k, F, eF in [('a', a0, ea0), ('b', b0, eb0), ('c', c0, ec0)]:
         x, e, f, ef = c[k][i], c['e' + k][i], F[0][B], eF[0][B]
         row.append('%+7.3f (%+4.1f sig)  ' % (x - f, (x - f)/np.hypot(e, ef)))
      print('  %s  %s %.3f/%.3f' % (B, ''.join(row), sig0[0][B], c['sig'][i]))
