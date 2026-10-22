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


def test_get_mangled_SED_H3L_uses_fractional_epoch(snobj):
   '''get_mangled_SED must pass the true (fractional) epoch for H3+L.

   Regression test: it truncated the epoch to int, silently returning the
   SED for a different day than the one used to compute the k-correction.
   '''
   import snpy
   snobj.EBVgal = 0.041
   snobj.k_version = 'H3+L'
   snobj.choose_model('EBV_model2', stype='st')
   snobj.fit(bands=['B', 'V'])

   band = next(b for b in snobj.data if b in snobj.ks_mopts)
   i = 0
   wave, man_flux, oflux, mfunc = snobj.get_mangled_SED(band, i, normalize=False)
   assert wave is not None

   # the true (fractional) epoch: what the SED must be evaluated at
   epoch = snobj.data[band].t[i]/(1+snobj.z)
   assert epoch != int(epoch)   # sanity: this epoch actually is fractional
   w_true, f_true = kcorr.get_SED(epoch, version='H3+L', sBV=snobj.ks_s)

   # get_mangled_SED returns wave*(1+z); compare in the observed frame
   f_true_on_wave = interp(wave, w_true*(1+snobj.z), f_true)
   # the returned original SED should match the fractional-epoch template,
   # i.e. NOT the floored one
   w_int, f_int = kcorr.get_SED(int(epoch), version='H3+L', sBV=snobj.ks_s)
   f_int_on_wave = interp(wave, w_int*(1+snobj.z), f_int)
   assert sum(abs(oflux - f_true_on_wave)) < sum(abs(oflux - f_int_on_wave))




def test_get_SED_H3L_extrapolates_out_of_range_epochs():
   '''With extrapolate=True, H3+L must clamp out-of-range epochs like the
   other SED versions instead of raising from the NIR template.'''
   for day, edge in [(75.0, 70.0), (-22.0, -19.0)]:
      w, f = kcorr.get_SED(day, version='H3+L', sBV=1.0, extrapolate=True)
      assert w is not None and all(isfinite(f))
      we, fe = kcorr.get_SED(edge, version='H3+L', sBV=1.0, extrapolate=True)
      assert allclose(f, fe)
   # without extrapolation they are simply undefined
   assert kcorr.get_SED(75.0, version='H3+L', sBV=1.0)[0] is None
   assert kcorr.get_SED_H3_plus_L(75.0, 1.0)[0] is None
