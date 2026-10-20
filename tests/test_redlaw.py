'''Tests for the independent host-galaxy and Milky-Way extinction laws.

The MW (foreground) law is ``sn.redlaw_gal`` and the host law is
``sn.redlaw``; both default to ``'ccm'``.  These tests pin that default and
verify that each law actually affects the result independently.
'''
import numpy as np
import pytest
import os
from snpy import get_sn, kcorr, fset, getSED


def test_redlaw_defaults(snobj):
   '''Both extinction-law attributes default to ccm.'''
   assert snobj.redlaw == 'ccm'
   assert snobj.redlaw_gal == 'ccm'


def test_redlaw_settable(snobj):
   '''The laws are settable attributes (via __setattr__).'''
   snobj.redlaw = 'fm'
   snobj.redlaw_gal = 'fm07'
   assert snobj.redlaw == 'fm'
   assert snobj.redlaw_gal == 'fm07'


def test_redden_default_equals_explicit_ccm():
   '''redden() with no law must match an explicit ccm/ccm call.'''
   wave, flux = getSED(0, version="H3")
   d = kcorr.redden(wave, flux, 0.1, 0.05, 0.02)
   e = kcorr.redden(wave, flux, 0.1, 0.05, 0.02,
                    redlaw_gal='ccm', redlaw_host='ccm')
   assert np.allclose(d, e)


def test_redden_independent_laws():
   '''Changing only the MW law, or only the host law, changes the output.

   Extinction is multiplicative in flux, so the two components cannot be
   applied separately without changing the result: this checks the codes
   threads each law to the correct component.
   '''
   wave, flux = getSED(0, version="H3")
   base = kcorr.redden(wave, flux, 0.1, 0.05, 0.0,
                       redlaw_gal='ccm', redlaw_host='ccm')
   gal = kcorr.redden(wave, flux, 0.1, 0.05, 0.0,
                      redlaw_gal='fm', redlaw_host='ccm')
   host = kcorr.redden(wave, flux, 0.1, 0.05, 0.0,
                       redlaw_gal='ccm', redlaw_host='fm')
   # use relative differences: the absolute scale is tiny
   assert np.max(np.abs((base - gal) / base)) > 1e-3
   assert np.max(np.abs((base - host) / base)) > 1e-3


def test_filter_R_laws():
   '''filter.R() differs between ccm and fm, and defaults to ccm.'''
   # B and V ratios for ccm (reference values)
   shB = fset['B'].R(Rv=3.1, redlaw='ccm')
   shV = fset['V'].R(Rv=3.1, redlaw='ccm')
   assert abs(shB - 4.088938) < 1e-4
   assert abs(shV - 3.226999) < 1e-4
   # fm gives different ratios
   fmB = fset['B'].R(Rv=3.1, redlaw='fm')
   assert abs(fmB - 4.055629) < 1e-4
   assert not np.isclose(shB, fmB)


def _fit_with_gal_law(datadir, law):
   import snpy
   sn = snpy.get_sn(os.path.join(datadir, 'SN2006ax.txt'))
   sn.EBVgal = 0.041
   sn.redlaw_gal = law
   sn.choose_model('EBV_model2', stype='st')
   sn.fit(bands=['B', 'V'], dokcorr=False)
   return {k: float(v) for k, v in sn.parameters.items()}


def test_gal_law_affects_fit(datadir):
   '''Changing only the MW dust law changes an EBV_model2 fit.

   Regression test for the MW law not being threaded into the Robs table
   that the fitter uses, which made redlaw_gal have no effect.
   '''
   ccm = _fit_with_gal_law(datadir, 'ccm')
   fm = _fit_with_gal_law(datadir, 'fm')
   assert ccm['DM'] != fm['DM']
   assert ccm['EBVhost'] != fm['EBVhost']
