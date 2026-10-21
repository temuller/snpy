#!/usr/bin/env python

# This script creates a fake SN spectrum, puts it at several redshifts
# and compares with kcorr

import pytest
from snpy import kcorr,getSED,fset,mangle_spectrum
from numpy import *

def test_kcorr():
   filters = ['u','B','V','r','i','Y','J','H']

   wave0,flux0 = getSED(0, version="H3")
   # redden the flux a bit
   rflux = kcorr.redden(wave0,flux0, 0.1, 0.0, 0.0)
   # generate artificial photometry at redshift 0.1:
   mag0s = []
   magzs = []
   for f in filters:
      mag0s.append(fset[f].synth_mag(wave0, rflux, z=0.))
      magzs.append(fset[f].synth_mag(wave0, rflux, z=0.1)+2.5*log10(1.1))
   
   mag0s = array([mag0s])
   magzs = array([magzs])
   masks = greater(magzs, 0)
   k0s = array([kcorr.kcorr([0], f, f, 0.1)[0][0] for f in filters])
   ks,mask,Rts,m_opts = kcorr.kcorr_mangle([0], filters, magzs, masks, filters, 
         0.1, version='H3', full_output=True, gradient=True, normfilter='H')
   # difference in kcorrs with no mangling:
   delta1 = absolute((magzs[0,:]-mag0s)[0] - k0s)
   # difference in kcorrs with mangling:
   delta2 = absolute((magzs[0,:]-mag0s)[0] - ks[0])

   # The mangling should be better and precise
   assert all(delta2 < delta1) and all(delta2 < 0.01)
   
def test_SEDs():
   wave1,flux1 = kcorr.get_SED(0, version='H3')
   wave2,flux2 = kcorr.get_SED(0, version='H3+L')

   delta1 = fset['B'].synth_mag(wave1, flux1) - fset['B'].synth_mag(wave2,flux2)
   delta2 = fset['H'].synth_mag(wave1, flux1) - fset['H'].synth_mag(wave2,flux2)

   assert (delta1 < 0.001) and (delta2 - 0.785952 < 0.001)


def test_A_obs_out_of_range_sentinel():
   '''A_obs must return the 99.9 sentinel, not raise, for out-of-range epochs.

   Regression test: the guard used the wrong accumulator name (Rs instead of
   As), so an epoch outside the template range raised NameError.
   '''
   # the 91bg template is only defined over (-13, 100); -18 stays below that
   # even after A_obs's own clamping, so get_SED returns (None, None).
   a = kcorr.A_obs('B', 0.01, [-18.], 0.05, 0.01, version='91bg')
   assert array(a)[0] == 99.9
   # valid epochs still return finite absorption
   a2 = kcorr.A_obs('B', 0.01, [0., 20.], 0.05, 0.01, version='H3')
   assert all(isfinite(a2)) and all(array(a2) < 90)


def test_kcorr_mangle_H3L_multiday():
   '''kcorr_mangle over several epochs with the H3+L template.

   Regression test: the multi-epoch day-selection used days[i] (an
   undefined name) instead of days[j] for H3+L.
   '''
   days = [0., 5., 10.]
   filts = ['B', 'V']
   mags = array([[15.0, 14.5], [15.1, 14.6], [15.3, 14.8]])
   mmask = ones_like(mags, dtype=bool)
   K, mask = kcorr.kcorr_mangle(days, filts, mags, mmask, ['B', 'V'], 0.01,
         sBV=1.0, version='H3+L')
   K = array(K)
   assert K.shape == (3, 2)
   assert all(isfinite(K))


