'''Step 3 of the EBV_model2 calibration: write the six variants from
calibrate.py as a calibration file in the format of st_calibration2.dat
(read by snpy.model.read_table):

   # <description>
   <variant> <band> a ea b eb c ec Rv eRv sigma

Usage:
   python write_calfile.py results/cal{0,1,2,3,4,5}.json --out st_calibration3.dat
'''
import json, argparse

if __name__ == '__main__':
   ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
   ap.add_argument('cals', nargs='+')
   ap.add_argument('--out', required=True)
   a = ap.parse_args()
   cals = sorted((json.load(open(f)) for f in a.cals), key=lambda c: c['variant'])
   lines = []
   for c in cals:
      lines.append('# %s\n' % c['description'])
      for i, b in enumerate(c['bands']):
         lines.append('%d %s %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n' % (
            c['variant'], b, c['a'][i], c['ea'][i], c['b'][i], c['eb'][i], c['c'][i], c['ec'][i],
            c['Rv'], c['eRv'], c['sig'][i]))
   open(a.out, 'w').writelines(lines)
   print('wrote %s: variants %s' % (a.out, [c['variant'] for c in cals]))
