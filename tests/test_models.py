import pytest
from snpy import get_sn
import numpy as num
import os

# Test if SALT2 is available
havesalt = pytest.mark.skipif('SALTPATH' not in os.environ,
      reason="SALTPATH environment variable not set (SALT2 not installed?)")

havemlcs = pytest.mark.skipif('MLCS2K2_BASEDIR' not in os.environ,
    reason="MLCS2K2_BASEDIR environment variable not set (MLCS not installed?")

@pytest.fixture
def snobj(datadir):
   import snpy
   snobj = snpy.get_sn(os.path.join(datadir, 'SN2006ax.txt'))
   return snobj

# Test fitting all the models. First, here are the results we expect

results = {
      'EBV_model':
          {'DM': 34.159,
           'EBVhost': -0.003,
           'Tmax': 827.681,
           'dm15': 1.086},
      'EBV_model2':
           {'DM': 34.254,
            'EBVhost': 0.0091,
            'Tmax': 827.570,
            'st': 0.963},
      'max_model':
           {'Bmax': 15.001,
            'Hmax': 15.991,
            'Jmax': 15.641,
            'Tmax': 827.553,
            'Vmax': 15.069,
            'Ymax': 15.757,
            'gmax': 14.957,
            'imax': 15.695,
            'rmax': 15.123,
            'st': 0.970,
            'umax': 15.399},
      'color_model':
           {'Bmax': 14.834,
            'EBVhost': 0.040,
            'Rv': 3.1,
            'Tmax': 827.571,
            'st': 0.963},
}
SALT_results = {
            'X0':0.018,
            'X1':0.121,
            'Color':-0.105,
            'Tmax':827.981,
            'Bmax':15.011,
}

MLCS_results = {
      'del':-0.107,
      'av0':0.099,
      'DM':34.548,
      'Tmax':827.31,
      'Vmax':15.084
}

@pytest.mark.parametrize("model,result", list(results.items()))
def test_model(snobj, model, result):
   if model == 'EBV_model':  
      stype='dm15'
   else:
      stype = 'st'
     
   snobj.replot=False
   snobj.choose_model(model, stype=stype)
   if model == 'color_model':
      snobj.fit(dokcorr=False, Rv=3.1)
   else:
      snobj.fit(dokcorr=False)

   res = [round(snobj.model.parameters[key],3) == round(result[key],3) \
         for key in result]
   assert num.all(res)

def test_model_fit_zero_weight_error(snobj):
   """A model whose mask is everywhere zero must give an actionable error.

   Reproduces the cryptic "All weights for filter B are zero" failure that
   the 'st' shape parameter can hit for objects outside the calibration
   sample, and checks that the raised message names the shape parameter and
   suggests an alternative instead of a bare RuntimeError.
   """
   snobj.replot = False
   snobj.choose_model('max_model', stype='st')

   def _all_zero_mask(band, MJD, *a, **k):
      n = len(MJD)
      return (num.zeros(n), num.zeros(n), num.zeros(n, dtype=bool))

   snobj.model.__call__ = _all_zero_mask
   with pytest.raises(RuntimeError, match="shape parameter"):
      snobj.fit(bands=['B', 'V'])


@havesalt
def test_SALT(snobj):
   snobj.replot=False
   snobj.choose_model('SALT_model', workdir='temp')
   snobj.fit()

   res = [round(snobj.model.parameters[key],3) == round(SALT_results[key],3) \
         for key in SALT_results]
   assert num.all(res)

@havemlcs
def test_MLCS(snobj):
   snobj.replot=False
   snobj.choose_model('MLCS_model')
   snobj.fit()

   res = [round(snobj.model.parameters[key],3) == round(MLCS_results[key],3) \
         for key in MLCS_results]
   assert num.all(res)


def test_max_model_extra_error_uses_nearest_epoch(snobj):
   """The late-start extra error must come from the table row closest to
   the first observed epoch, not the farthest one."""
   snobj.replot = False
   snobj.choose_model('max_model', stype='st')
   snobj.fit(dokcorr=False)
   m = snobj.model
   first = num.median([snobj.data[b].MJD.min() for b in snobj.data
                       if snobj.restbands[b]+'max' in m.parameters])
   # days:            0     5     10    15    20
   # Tmax extra err:  0.07  0.16  0.21  0.21  0.46
   for t0, expected in [(1.0, 0.07), (8.0, 0.21), (19.0, 0.46)]:
      m.parameters['Tmax'] = first - t0*(1 + snobj.z)
      assert m._extra_error('Tmax') == pytest.approx(expected)


def test_extra_error_added_in_quadrature_to_free_params_only(snobj):
   snobj.replot = False
   snobj.choose_model('max_model', stype='st')
   snobj.model._extra_error = lambda p: 0.1
   snobj.fit(dokcorr=False, Tmax=827.55)
   m = snobj.model
   assert m.errors['Tmax'] == 0
   for p in m._free:
      assert m.errors[p] == pytest.approx(num.sqrt(0.1**2 + m.C[p][p]))


@pytest.mark.parametrize("st,expected", [(1.35, 1.3), (0.35, 0.4)])
def test_H3L_kcorr_clamps_stretch_to_SED_range(snobj, st, expected):
   """An 'st' outside the H3+L sBV range must not crash the k-corrections."""
   snobj.replot = False
   snobj.k_version = 'H3+L'
   snobj.choose_model('max_model', stype='st')
   snobj.fit(bands=['B', 'V'], dokcorr=False, st=st)
   snobj.kcorr(bands=['B', 'V'], mangle=0)
   assert snobj.ks_s == expected
   assert num.all(num.isfinite(snobj.ks['B']))


def test_kcorr_stretch_comes_from_fitted_st(snobj):
   """A dm15 from a B-band light-curve fit must not override the model's
   fitted 'st' when choosing the stretch for the k-corrections."""
   snobj.replot = False
   snobj.choose_model('max_model', stype='st')
   snobj.fit(bands=['B', 'V'], dokcorr=False, st=0.9)
   snobj.data['B'].dm15 = 1.6     # as left behind by e.g. lc.template()
   snobj.kcorr(bands=['B', 'V'], mangle=0)
   assert snobj.ks_s == 0.9


@pytest.mark.parametrize("fit_args,calibration", [({}, 0), ({'calibration': 1}, 1)])
def test_EBV_model2_systematics_uses_fit_calibration(snobj, fit_args, calibration):
   snobj.replot = False
   snobj.choose_model('EBV_model2', stype='st')
   snobj.fit(dokcorr=False, **fit_args)
   other = 1 - calibration
   DM = snobj.systematics()['DM']
   assert DM == pytest.approx(snobj.systematics(calibration=calibration)['DM'])
   assert DM != pytest.approx(snobj.systematics(calibration=other)['DM'])


@pytest.mark.parametrize("kind,param", [("stemplate", 0.9), ("template", 1.3)])
@pytest.mark.parametrize("band", ['u', 'i', 'H'])
def test_template_offset_is_time_dilated(kind, param, band):
   """The band's Tmax - Tmax(B) offset is a rest-frame quantity, so the
   redshifted template must be the rest-frame one stretched by (1+z)."""
   from snpy import ubertemp
   temp = getattr(ubertemp, kind)()
   temp.mktemplate(param)
   z = 0.3
   t = num.linspace(-8., 30., 40)
   m0, e0, mask0 = temp.eval(band, t, 0, gen=2)
   mz, ez, maskz = temp.eval(band, t*(1+z), z, gen=2)
   assert num.array_equal(mask0, maskz)
   assert num.allclose(m0[mask0], mz[maskz], rtol=0, atol=1e-10)


def test_st_template_gen3_uses_gen3_Tmax_offsets():
   from snpy import ubertemp
   from snpy.CSPtemp import Tmax_splines
   temp = ubertemp.stemplate()
   temp.mktemplate(0.9)
   dT = float(Tmax_splines['i'](0.9))
   assert temp.Ct.deltaTmax('i', 3) == pytest.approx(dT)
   t = num.linspace(-5., 30., 30)
   m1 = temp.eval('i', t, gen=3)[0]
   m2 = temp.eval('i', t - dT, gen=3, toff=False)[0]
   assert num.allclose(m1, m2)
   assert temp.Ct.deltaTmax('B', 3) == 0


@pytest.mark.parametrize("model", ['max_model', 'EBV_model2'])
def test_get_max_Tmax_is_in_observer_frame(snobj, model):
   """get_max must convert the rest-frame template peak to observed days."""
   snobj.replot = False
   snobj.choose_model(model, stype='st')
   snobj.fit(dokcorr=False)
   m = snobj.model
   snobj.z = 0.3      # exaggerate time dilation
   Tmaxs, Mmaxs, eMmaxs, rbands = m.get_max(['i'])
   # numerically locate the observed-frame peak of the model light curve
   t = m.Tmax + num.linspace(-15, 15, 30001)
   m.template.mktemplate(m.parameters[m.stype])
   mag = m.template.eval('i', t - m.Tmax, snobj.z, gen=m.gen)[0]
   assert Tmaxs[0] == pytest.approx(t[num.argmin(mag)], abs=2e-3)
