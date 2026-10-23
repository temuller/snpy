'''Step 1 of the EBV_model2 calibration: fit the calibration sample with
max_model and store the peak magnitudes, st and their covariance.

Usage:
   python fit_peaks.py --sample 2012 --gen 3.1 --kversion H3+L --out results/fits.json

Samples:
   2012      the CSP-I SNe Ia of data releases 1 and 2 (Contreras+2010,
             Stritzinger+2011): what was available for st_calibration2.dat.
   extended  every CSP-I and CSP-II SN Ia in the SAINTS compilation.

Peculiar SNe are flagged (not removed) from the 'Type' column of the SAINTS
sample table; see PECULIAR_TYPES.
'''
import os, sys, json, io, argparse, contextlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', '..', 'saints', 'saints_photometry')

DR1 = ('2004dt 2004ef 2004eo 2004ey 2004gc 2004gs 2004gu 2005A 2005M 2005W 2005ag 2005al 2005am '
       '2005be 2005bg 2005bl 2005bo 2005el 2005eq 2005hc 2005iq 2005ir 2005kc 2005ke 2005ki 2005lu '
       '2005na 2006D 2006X 2006ax 2006bh 2006eq 2006gt 2006mr 2006py').split()
DR2 = ('2005hj 2005ku 2005mc 2006bd 2006br 2006bt 2006ef 2006ej 2006et 2006ev 2006fw 2006gj 2006hb '
       '2006hx 2006is 2006kf 2006lu 2006ob 2006os 2006ot 2007A 2007N 2007S 2007af 2007ai 2007as '
       '2007ax 2007ba 2007bc 2007bd 2007bm 2007ca 2007hx 2007if 2007jg 2007jh 2007le 2007mm 2007nq '
       '2007on 2008C 2008R 2008bc 2008bq 2008fp 2008gp 2008hv 2008ia 2009F 2009dc').split()
# Normal, 91T-like and 91bg-like SNe are kept (st cuts deal with fast decliners)
PECULIAR_TYPES = ['SNIa-pec', 'SNIax', 'SNIa-SC', 'SNIa-06bt', 'SNIa-02es', 'SNIa-06gz', 'transitional']
RESTBANDS = ['u', 'B', 'V', 'g', 'r', 'i', 'Y', 'J', 'H']


def sample(name):
   import csv
   rows = {r['SN']: r for r in csv.DictReader(open(os.path.join(DATA_DIR, 'saints_sample.csv')))
           if r['survey'] == 'CSP'}
   if name == '2012':
      names = ['SN' + n for n in DR1 + DR2]
   elif name == 'extended':
      names = sorted(rows)
   else:
      raise ValueError("sample must be '2012' or 'extended'")
   return [(n, rows[n]['Type'] if n in rows else None) for n in names]


def init():
   global snpy
   import warnings; warnings.filterwarnings('ignore')
   import snpy as _snpy; snpy = _snpy


def fit(name, sntype, gen, kversion):
   out = {'name': name, 'type': sntype, 'peculiar': sntype in PECULIAR_TYPES}
   f = os.path.join(DATA_DIR, 'data', '%s_CSP.txt' % name)
   if sntype is None or not os.path.exists(f):
      out['error'] = 'no data'
      return out
   try:
      with contextlib.redirect_stdout(io.StringIO()):
         s = snpy.get_sn(f)
         s.replot = 0; s.quiet = 1
         s.k_version = kversion; s.redlaw = 'ccm'; s.redlaw_gal = 'ccm'
         out.update(z=float(s.z), zcmb=float(s.zcmb), EBVgal=float(s.EBVgal))
         s.choose_model('max_model', stype='st')
         bands = [b for b in s.data if s.restbands[b] in RESTBANDS]
         s.fit(bands=bands, gen=gen)
      m = s.model
      out.update(bands=bands, ks_s=float(s.ks_s),
                 params={k: float(v) for k, v in m.parameters.items()},
                 errors={k: float(v) for k, v in m.errors.items()},
                 C={p: {q: float(m.C[p][q]) for q in m.C[p]} for p in m.C},
                 rchisq=float(m.rchisquare), dof=int(m.dof))
   except Exception as e:
      out['error'] = '%s: %s' % (type(e).__name__, str(e)[:150])
   return out


if __name__ == '__main__':
   ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
   ap.add_argument('--sample', default='2012')
   ap.add_argument('--gen', type=float, default=3.1)
   ap.add_argument('--kversion', default='H3+L')
   ap.add_argument('--nproc', type=int, default=8)
   ap.add_argument('--out', required=True)
   a = ap.parse_args()
   gen = int(a.gen) if a.gen == int(a.gen) else a.gen
   sne = sample(a.sample)
   with ProcessPoolExecutor(a.nproc, mp_context=mp.get_context('spawn'), initializer=init) as ex:
      res = list(ex.map(fit, [n for n, t in sne], [t for n, t in sne],
                        [gen]*len(sne), [a.kversion]*len(sne)))
   meta = {'sample': a.sample, 'gen': gen, 'kversion': a.kversion, 'redlaw': 'ccm'}
   json.dump({'meta': meta, 'fits': res}, open(a.out, 'w'))
   bad = [(r['name'], r['error'][:40]) for r in res if 'error' in r]
   print('%d SNe: %d fitted, %d without data or failed: %s' % (len(res), len(res) - len(bad), len(bad), bad))
