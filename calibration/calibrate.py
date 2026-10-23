'''Step 2 of the EBV_model2 calibration: fit the peak absolute magnitudes,
their dependence on st, the host R_V and the per-band intrinsic scatter on
the max_model peaks from fit_peaks.py (Burns+2011 "color excess" model):

   m_Xj = a_X + b_X (st_j - 1) + c_X (st_j - 1)^2 + R_X(R_V) E_j + mu_j

with E_j >= 0 and the Folatelli+2010 low-reddening SNe anchored at E_j = 0
(the Burns+2011 "blue prior"). R_X is computed as EBV_model2 does (R_obs of the
SED at maximum for E(B-V) = 0.01 and the CCM+O'Donnell law); mu_j uses
H0 = 72 and Omega_M = 0.28 with CMB-frame redshifts. The st errors (and their
covariance with the peaks) are propagated through dM/dst. E_j is solved
analytically for given global parameters, which maximize the exact likelihood.
Errors are half the 16-84% range of a bootstrap over the SNe.

This reproduces calibration 0 of st_calibration2.dat within 2 sigma when run
on the 2012 sample with gen=2 templates and the H3 SED (see README.md).

Usage:
   python calibrate.py results/fits.json --variant 0 --nboot 100 --out results/cal0.json

Variants (as in st_calibration2.dat): 0/1/2 = with u-band, and st > 0.5 /
(B-V)max < 0.3 / all objects; 3/4/5 = the same without u-band.
'''
import json, argparse, warnings
import numpy as np
from scipy.optimize import minimize
from astropy.cosmology import FlatLambdaCDM
warnings.filterwarnings('ignore')

F10_LOW_REDDENING = ['2004eo', '2005M', '2005al', '2005am', '2005el', '2005hc', '2005iq', '2005ke',
                     '2005ki', '2006bh', '2004ey', '2005ag', '2006D', '2006ax', '2006gt']
RV_GRID = np.arange(1.0, 4.01, 0.1)
BIG = 1e6      # variance given to missing bands
VARIANTS = {0: 'u-band, st > 0.5', 1: 'u-band, (B-V) < 0.3', 2: 'u-band, all objects',
            3: 'no u-band, st > 0.5', 4: 'no u-band, (B-V) < 0.3', 5: 'no u-band, all objects'}


def load(fitfile, variant, sig_v=0., zmin=0.01):
   from snpy import kcorr
   F = json.load(open(fitfile))
   kver = F['meta']['kversion']
   bands_all = ['B', 'V', 'u', 'g', 'r', 'i', 'Y', 'J', 'H'] if variant < 3 else \
               ['B', 'V', 'g', 'r', 'i', 'Y', 'J', 'H']
   cosmo = FlatLambdaCDM(H0=72, Om0=0.28)
   sne = []
   for r in F['fits']:
      if 'error' in r or r['peculiar'] or r['zcmb'] <= zmin:
         continue
      p = r['params']; st = p['st']
      BV = p['Bmax'] - p['Vmax'] if 'Bmax' in p and 'Vmax' in p else np.nan
      if variant % 3 == 0 and not st > 0.5: continue
      if variant % 3 == 1 and not BV < 0.3: continue
      bands = [b for b in bands_all if b + 'max' in p]
      keys = ['st'] + [b + 'max' for b in bands]
      C = np.array([[r['C'][i][j] for j in keys] for i in keys])
      mu = cosmo.distmod(r['zcmb']).value + 5*np.log10((1 + r['z'])/(1 + r['zcmb']))
      sBV = min(max(st, 0.4), 1.3)
      Rg = np.array([[kcorr.R_obs(b, r['z'], 0, 0.01, 0, Rv, 3.1, kver, redlaw='ccm', sBV=sBV)
                      for Rv in RV_GRID] for b in bands])
      sne.append(dict(name=r['name'], d=st - 1, m=np.array([p[b + 'max'] for b in bands]),
                      idx=np.array([bands_all.index(b) for b in bands]), Cst=C[0, 0],
                      Cms=C[1:, 0], Cmm=C[1:, 1:], mu=mu,
                      smu=5/np.log(10)*sig_v/(2.998e5*r['zcmb']), Rg=Rg,
                      anchor=r['name'].replace('SN', '') in F10_LOW_REDDENING))
   return bands_all, sne, F['meta']


def stack(sample, NB):
   '''Pad every SN to all bands (missing ones get a huge variance).'''
   N = len(sample)
   T = dict(d=np.array([s['d'] for s in sample]), mu=np.array([s['mu'] for s in sample]),
            smu=np.array([s['smu'] for s in sample]), Cst=np.array([s['Cst'] for s in sample]),
            anchor=np.array([s['anchor'] for s in sample]),
            m=np.zeros((N, NB)), Cms=np.zeros((N, NB)), Cmm=np.zeros((N, NB, NB)),
            have=np.zeros((N, NB), bool), Rg=np.zeros((N, NB, len(RV_GRID))))
   for j, s in enumerate(sample):
      i = s['idx']
      T['m'][j, i] = s['m']; T['Cms'][j, i] = s['Cms']; T['have'][j, i] = True
      T['Cmm'][j][np.ix_(i, i)] = s['Cmm']; T['Rg'][j, i] = s['Rg']
   return T


def nll(th, T, NB, retE=False):
   a, b, c, sig, Rv = th[:NB], th[NB:2*NB], th[2*NB:3*NB], np.exp(th[3*NB:4*NB]), th[-1]
   d, h = T['d'][:, None], T['have']
   g = (b + 2*c*d)*h            # dM/dst
   Sig = T['Cmm'] + g[:, :, None]*g[:, None, :]*T['Cst'][:, None, None] \
         - g[:, :, None]*T['Cms'][:, None, :] - T['Cms'][:, :, None]*g[:, None, :] \
         + (T['smu']**2)[:, None, None]*(h[:, :, None] & h[:, None, :])
   Sig[:, np.arange(NB), np.arange(NB)] += np.where(h, sig**2, BIG)
   W = np.linalg.inv(Sig)
   k = min(max(np.searchsorted(RV_GRID, Rv) - 1, 0), len(RV_GRID) - 2)
   w = (Rv - RV_GRID[k])/(RV_GRID[1] - RV_GRID[0])
   Rx = (T['Rg'][:, :, k]*(1 - w) + T['Rg'][:, :, k + 1]*w)*h
   r0 = np.where(h, T['m'] - (a + b*d + c*d*d) - T['mu'][:, None], 0)
   WR = np.einsum('nij,nj->ni', W, Rx)
   E = np.maximum(0, np.einsum('ni,ni->n', WR, r0)/np.einsum('ni,ni->n', WR, Rx))
   E = np.where(T['anchor'], 0, E)
   res = r0 - Rx*E[:, None]
   tot = 0.5*np.einsum('ni,nij,nj->', res, W, res) + 0.5*np.sum(np.linalg.slogdet(Sig)[1])
   return (tot, E) if retE else tot


def fit(sample, NB, th0=None):
   T = stack(sample, NB)
   if th0 is None:
      th0 = np.r_[np.full(NB, -19.), np.full(NB, -0.6), np.full(NB, 1.), np.full(NB, np.log(0.08)), 1.7]
   bnds = [(None, None)]*(3*NB) + [(-7, 0)]*NB + [(RV_GRID[0], RV_GRID[-1])]
   best = None
   for _ in range(3):            # restart from the solution until it stops improving
      res = minimize(nll, th0, args=(T, NB), method='L-BFGS-B', bounds=bnds,
                     options={'maxfun': 200000, 'maxiter': 20000})
      done = best is not None and abs(res.fun - best.fun) < 1e-6
      best = res; th0 = res.x
      if done: break
   return best.x, nll(best.x, T, NB, retE=True)[1], best.success


def _boot(args):
   seed, sne, NB, th0 = args
   rng = np.random.default_rng(seed)
   th = fit([sne[i] for i in rng.integers(0, len(sne), len(sne))], NB, th0)[0]
   return np.r_[th[:3*NB], np.exp(th[3*NB:4*NB]), th[-1]]


if __name__ == '__main__':
   import multiprocessing as mp
   ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
   ap.add_argument('fits')
   ap.add_argument('--variant', type=int, default=0, choices=sorted(VARIANTS))
   ap.add_argument('--sig-v', type=float, default=0., help='peculiar velocities (km/s); 0 = per-band scatter only')
   ap.add_argument('--nboot', type=int, default=100)
   ap.add_argument('--nproc', type=int, default=8)
   ap.add_argument('--out', required=True)
   a = ap.parse_args()
   bands, sne, meta = load(a.fits, a.variant, a.sig_v)
   NB = len(bands)
   print('variant %d (%s): %d SNe, %d anchored at E=0' % (a.variant, VARIANTS[a.variant], len(sne),
         sum(s['anchor'] for s in sne)), flush=True)
   th, E, ok = fit(sne, NB)
   boots = []
   if a.nboot > 0:
      with mp.get_context('fork').Pool(a.nproc) as pool:
         for k, b in enumerate(pool.imap_unordered(_boot, [(i, sne, NB, th) for i in range(a.nboot)])):
            boots.append(b)
            if (k + 1) % 10 == 0: print('  bootstrap %d/%d' % (k + 1, a.nboot), flush=True)
   # Robust errors: a few resamples that miss most fast decliners leave the
   # quadratic term unconstrained and give heavy tails, which would dominate
   # a standard deviation.
   if boots:
      p16, p84 = np.percentile(boots, [16, 84], axis=0)
      err = (p84 - p16)/2
   else:
      err = np.zeros(4*NB + 1)
   out = dict(meta, variant=a.variant, description=VARIANTS[a.variant], N=len(sne), sig_v=a.sig_v,
              bands=bands, converged=bool(ok), nboot=a.nboot, sne=[s['name'] for s in sne],
              a=th[:NB].tolist(), b=th[NB:2*NB].tolist(), c=th[2*NB:3*NB].tolist(),
              sig=np.exp(th[3*NB:4*NB]).tolist(), Rv=float(th[-1]),
              ea=err[:NB].tolist(), eb=err[NB:2*NB].tolist(), ec=err[2*NB:3*NB].tolist(),
              esig=err[3*NB:4*NB].tolist(), eRv=float(err[-1]), boots=np.array(boots).tolist(),
              E=dict(zip([s['name'] for s in sne], E.tolist())))
   json.dump(out, open(a.out, 'w'))
   print('converged: %s   R_V = %.3f +- %.3f' % (ok, out['Rv'], out['eRv']))
   for i, B in enumerate(bands):
      print('%s  a=%.3f(%.3f)  b=%.3f(%.3f)  c=%.3f(%.3f)  sig=%.3f(%.3f)' % (B, out['a'][i], out['ea'][i],
            out['b'][i], out['eb'][i], out['c'][i], out['ec'][i], out['sig'][i], out['esig'][i]))
