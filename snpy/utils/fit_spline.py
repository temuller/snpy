'''This module provides some helper functions for fitting a spline to Supernova
data.'''

from scipy.interpolate import splrep
from scipy.interpolate import splev
from scipy.integrate import quad
import numpy as np
import sys,os,string
from snpy import spline2
#import fit_lightcurves as lc
import scipy
import scipy.optimize


def make_spline(t, m, e_m, knots=None, k=1, s=None, fitflux=0, zpt=0, 
      tmin=None, tmax=None, task=-1, anchor_dist=[5.0,5.0], slopes=[None,None]):
   '''A wrapper around splrep that makes sure the independent variable is
   monotonic and non-repeating.  Required arguments:  time (t), magnitudes
   (m) and errors (e_m).  If knots are specified, use them (if task==-1), otherwise,
   they are computed from -10 days to 100 days in 10-day increments.  k is the 
   spline order (default 1) and s is the smoothing factor, as per splrep.  If fitflux
   is nonzero, convert magnitudes to flux (using provided zpt).  tmin and tmax should
   be set to the limits of the spline.'''
   # first, we make sure that t is monotonically increasing with no repeated
   # elements
   sids = np.argsort(t)
   tt = t[sids]      #num.take(t, sids)
   mm = m[sids]      # num.take(m, sids)
   ee_m = e_m[sids]  #num.take(e_m, sids)

   if tmin is None:
      tmin = t.min()
   if tmax is None:
      tmax = t.max()

   # here's some Numeric magic.  first, find where we have repeating x-values
   Nmatrix = np.equal(tt[:,np.newaxis], tt[np.newaxis,:])
   #val_matrix = mm[:,num.newaxis]*num.ones((len(mm),len(mm)))
   #e_matrix = ee_m[:,num.newaxis]*num.ones((len(mm),len(mm)))
   val_matrix = mm[:,np.newaxis]*Nmatrix
   e_matrix = ee_m[:,np.newaxis]*Nmatrix

   average = sum(val_matrix)/sum(Nmatrix)
   e_average = sum(e_matrix)/sum(Nmatrix)
   
   # at this point, average is the original data, but with repeating data points
   # replaced with their average.  Now, we just pick out the unique x's and
   # the first of any repeating points:
   gids = np.concatenate([[1], np.greater(tt[1:] - tt[:-1], 0.)])
   tt = np.compress(gids, tt)
   mm = np.compress(gids, average)
   ee_m = np.compress(gids, e_average)

   # Now get rid of any data that's outside [tmin,tmax]
   gids = np.less_equal(tt, tmax)*np.greater_equal(tt, tmin)
   #tt = num.compress(gids,tt)
   tt = tt[gids]
   #mm = num.compress(gids,mm)
   mm = mm[gids]
   #ee_m = num.compress(gids,ee_m)
   ee_m = ee_m[gids]
   ee_m = np.where(np.less(ee_m, 0.001), 0.001, ee_m)

   # Next, add some anchors to the data to control the slopes
   if anchor_dist[0] > 0 and task != -1:
      if slopes[0] is not None:
         mm0 = mm[0] - slopes[0]*anchor_dist[0]
      else:
         mm0 = mm[0] - (mm[1] - mm[0])/(tt[1]-tt[0])*anchor_dist[0]
      tt = np.concatenate([[tt[0]-anchor_dist[0]], tt])
      mm = np.concatenate([[mm0], mm])
      ee_m = np.concatenate([[ee_m[0]], ee_m])
   if anchor_dist[1] > 0:
      if slopes[1] is not None:
         mm1 = mm[-1] + slopes[1]*anchor_dist[1]
      else:
         mm1 = mm[-1] + (mm[-1] - mm[-2])/(tt[-1]-tt[-2])*anchor_dist[1]
      tt = np.concatenate([tt,[tt[-1]+anchor_dist[1]]])
      mm = np.concatenate([mm,[mm1]])
      ee_m = np.concatenate([ee_m,[ee_m[-1]]])

   # Now convert to flux if requested:
   if fitflux:
      mm = np.power(10, -0.4*(mm - zpt))
      ee_m = mm*ee_m/1.087
   
   if knots is None and task==-1:
      # scipy passes knots given with task=-1 to FITPACK as *interior* knots
      # (it adds the k+1 boundary knots itself), so they must lie strictly
      # inside (tmin,tmax) and there can be at most len(tt)-k-1 of them.
      nint = min(2*k+1, len(tt)-k-1)
      if nint < 1:
         raise ValueError("Not enough data points to fit a spline of order %d"
               % k)
      knots = tmin + (tmax - tmin)*np.arange(1, nint+1)/(nint+1)
   
   # Okay, now make the spline representation
   tck,fp,ier,msg = splrep(tt, mm, 1.0/ee_m, k=k, s=s, t=knots, task=task,
         full_output=1)
   return(tck,fp,ier,msg)

def K2(x, tck):
   '''compute the square curvature of a spline2 at point x'''
   yp = np.power(spline2.evalsp(x, tck, 1),2)
   ypp = np.power(spline2.evalsp(x, tck, 2),2)
   res = np.asarray(ypp/np.power(1+yp,3))
   # spline2.evalsp returns a 1-element array for scalar input, but quad()
   # requires a plain scalar (numpy 2 will error on the conversion)
   return res.item() if res.size == 1 else res


def make_spline2(t, m, e_m, k=3, fitflux=0, zpt=0, tmin=-10, tmax=100,
      adaptive=0, max_curv_fac=10, **args):
   '''A wrapper around spline2 that makes sure the independent variable is
   monotonic and non-repeating.  Required arguments:  time (t), magnitudes
   (m) and errors (e_m).  k is the spline order (default 3)  If fitflux
   is nonzero, convert magnitudes to flux (using provided zpt).  tmin and tmax should
   be set to the limits of the spline.'''
   # first, we make sure that t is monotonically increasing with no repeated
   # elements
   sids = np.argsort(t)
   tt = np.take(t, sids)
   mm = np.take(m, sids)
   ee_m = np.take(e_m, sids)

   # here's some Numeric magic.  first, find where we have repeating x-values
   Nmatrix = np.equal(tt[:,np.newaxis], tt[np.newaxis,:])
   #val_matrix = mm[:,num.newaxis]*num.ones((len(mm),len(mm)))
   #e_matrix = ee_m[:,num.newaxis]*num.ones((len(mm),len(mm)))
   val_matrix = mm[:,np.newaxis]*Nmatrix
   e_matrix = ee_m[:,np.newaxis]*Nmatrix

   average = sum(val_matrix)/sum(Nmatrix)
   e_average = sum(e_matrix)/sum(Nmatrix)
   
   # at this point, average is the original data, but with repeating data points
   # replaced with their average.  Now, we just pick out the unique x's and
   # the first of any repeating points:
   gids = np.concatenate([[True], np.greater(tt[1:] - tt[:-1], 0.)])
   tt = tt[gids]           #num.compress(gids, tt)
   mm = average[gids]      # num.compress(gids, average)
   ee_m = e_average[gids]  #num.compress(gids, e_average)

   # Now get rid of any data that's outside [tmin,tmax]
   gids = np.less_equal(tt, tmax)*np.greater_equal(tt, tmin)
   tt = tt[gids]             #num.compress(gids,tt)
   mm = mm[gids]             # num.compress(gids,mm)
   ee_m = ee_m[gids]         #num.compress(gids,ee_m)
   ee_m = np.where(np.less(ee_m, 0.001), 0.001, ee_m)

   # Now convert to flux if requested:
   if fitflux:
      mm = np.power(10, -0.4*(mm - zpt))
      ee_m = mm*ee_m/1.087
   
   # Okay, now make the spline representation
   if not adaptive:
      tck = spline2.spline2(tt, mm, w=1.0/ee_m, degree=k, **args)
      fp = np.sum(np.power((mm - spline2.evalsp(tt, tck))/ee_m, 2))
      ier = 0
      msg = 'not much'
      return(tck,fp,ier,msg)
   
   # Do an adaptive (much slower) search for the best fit, subject
   #  to curvature constraints (right now, just a cap).
   if 'lopt' in args:  del args['lopt']
   Ks = []
   chisqs = []
   lopts = list(range(2, int(0.8*len(tt) - 1)))
   for l in lopts:
      tck = spline2.spline2(tt, mm, w=1.0/ee_m, degree=k, lopt=l, **args)
      fp = np.sum(np.power((mm - spline2.evalsp(tt, tck))/ee_m, 2))
      K = quad(K2, tck[0][0], tck[0][-1], args=(tck,), epsrel=0.01)[0]
      chisqs.append(fp)
      Ks.append(K)

   chisqs = np.array(chisqs)
   Ks = np.array(Ks)
   chisqs = np.where(np.less(Ks, Ks.min()*max_curv_fac), chisqs, np.inf)
   id = np.argmin(chisqs)

   ier = lopts[id]
   tck = spline2.spline2(tt, mm, w=1.0/ee_m, degree=k, lopt=ier, **args)
   fp = np.sum(np.power((mm - spline2.evalsp(tt, tck))/ee_m, 2))
   return(tck, fp, ier, "Optimized lopt = %d" % ier)

def fit_spline(t, m, e_m, knots=None, k=1, s=None, fitflux=0, zpt=0, tmin=-10, tmax=100,
      task=-1):
   '''Fit a spline to a set of lightcurve data (t,m,e_m).  The rest of the arguments are
   the same as make_spline() above.  The spline is then evaluated at one day intervals
   from t[0] to t[-1] and returned as a two-tuple (time,mag).'''

   tck,fp,ier,msg = make_spline(t, m, e_m, knots, k, s, fitflux, zpt, tmin, tmax,task)

   # Now do the interpolation
   evt = np.arange(t[0],t[-1]+1,1, dtype=np.float64)
   evm = splev(evt, tck)

   if fitflux:
      evm = -2.5*np.log10(evm) + zpt

   return(evt,evm)

def interp_spline(t, m, e_m, eval_t, knots=None, k=1, s=None, fitflux=0, zpt=0, 
      tmin=-10, tmax=100, task=-1, tol=0.1):
   '''Same as fit_spline, except that you decide on the interpolated times.  Good
   for making color estimates when data in different bands in at different epochs.
   In this case, only the interpolated magnitudes are returned.'''
   tck,fp,ier,msg = make_spline(t,m,e_m,knots,k,s,fitflux,zpt,tmin,tmax,task)
   evm = splev(eval_t, tck)

   # Now, we scan t and eval_t and find where they are less than tol.  In these
   #  cases, we take the average of any matching times
   delta = np.absolute(t[np.newaxis,:] - eval_t[:,np.newaxis])
   cond = np.less(delta, tol)
   values = np.array([m]*len(eval_t))*cond
   N = np.sum(cond, axis=1)
   w = np.where(N > 0, N, 1)
   s = np.sum(values, axis=1)
   evm = np.where(N > 0, s/w, evm)

   if fitflux:
      evm = -2.5*np.log10(evm) + zpt

   return(evm)

def find_extr(tck, numpoints=1000):

   xs = np.arange(numpoints+1, dtype='d')/numpoints*(tck[0][-1]- tck[0][0]) + tck[0][0]
   f = lambda x:  splev(x, tck, 1)
   derivs = f(xs)
   gt0 = np.greater(derivs, 0)
   inds = np.nonzero(np.logical_xor(gt0[1:], gt0[:-1]))[0]
   if len(inds) == 0:
      return (None,None,None)
   ret = []
   for i in range(len(inds)):
      ret.append(scipy.optimize.brentq(f, xs[inds[i]], xs[inds[i]+1]))
   ret = np.array(ret)
   vals = np.atleast_1d(splev(ret, tck))
   curvs = np.atleast_1d(splev(ret, tck, 2))
   curvs = np.where(np.greater(curvs, 0), 1, curvs)
   curvs = np.where(np.less(curvs, 0), -1, curvs)
   
   return (ret,vals,curvs)
