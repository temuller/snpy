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


@pytest.mark.parametrize("k_version", ['H3', 'H3+L'])
def test_kcorr_mask_uses_restframe_SED_epochs(snobj, k_version):
   """The model's k-correction mask must cut on the same rest-frame epochs
   (and SED limits) that sn.kcorr used, not on observer-frame days."""
   from snpy import kcorr
   snobj.replot = False
   snobj.k_version = k_version
   snobj.choose_model('max_model', stype='st')
   snobj.fit(bands=['B', 'V'], dokcorr=False)
   snobj.kcorr(bands=['B', 'V'], mangle=0)
   m = snobj.model
   snobj.z = 0.2      # exaggerate time dilation
   s = snobj.ks_s
   t = num.linspace(-30, 120, 1501)
   K, mask = m.kcorr('B', t)
   epoch = t/(1 + snobj.z)
   if k_version == 'H3+L':
      expected = (epoch >= -19) & (epoch <= 70) & (epoch/s > -19) & (epoch/s <= 70)
   else:
      expected = (epoch/s >= -19) & (epoch/s <= 70)
   # combined with the validity of the nearest data point's k-correction
   mids = num.argmin(num.absolute(t[:,None] - snobj.data['B'].MJD[None,:] + m.Tmax),
                     axis=1)
   expected &= snobj.ks_mask['B'][mids].astype(bool)
   assert num.array_equal(mask.astype(bool), expected)


@pytest.mark.parametrize("model,stype", [('EBV_model2', 'st'), ('EBV_model', 'dm15')])
def test_DM_systematic_weights_include_photometric_errors(snobj, model, stype):
   """Bands with much larger photometric errors must carry ~no weight in
   the averaged DM systematic."""
   snobj.replot = False
   snobj.choose_model(model, stype=stype)
   snobj.fit(dokcorr=False)
   m = snobj.model
   bands = m._fbands
   m._fbands = ['B']
   DM_B = snobj.systematics()['DM']
   m._fbands = bands
   assert snobj.systematics()['DM'] != pytest.approx(DM_B, rel=1e-3)
   for b in bands:
      if b != 'B':
         snobj.data[b].e_mag = snobj.data[b].e_mag*1e4
   assert snobj.systematics()['DM'] == pytest.approx(DM_B, rel=1e-3)


def _inject(model, band, idx, value, masked):
   """Make model(band) return `value` at data point `idx`."""
   call = model.__call__
   def wrapped(b, t, extrap=False):
      mod, err, mask = call(b, t, extrap)
      if b == band:
         mod = mod.copy(); mod[idx] = value
         mask = mask.copy().astype(bool); mask[idx] = not masked
      return mod, err, mask
   model.__call__ = wrapped


def test_absurd_model_values_at_masked_points_do_not_poison_fit(snobj):
   """A masked point whose model overflows (e.g. the K polynomial far
   outside its range) used to turn every residual into NaN."""
   snobj.replot = False
   snobj.data['V'].mask[-1] = False      # the reference fit ignores it too
   snobj.choose_model('max_model', stype='st')
   snobj.fit(bands=['B', 'V'], dokcorr=False)
   ref = dict(snobj.model.parameters)
   snobj.data['V'].mask[-1] = True
   snobj.choose_model('max_model', stype='st')
   _inject(snobj.model, 'V', -1, -900.0, masked=True)
   snobj.fit(bands=['B', 'V'], dokcorr=False)
   for p in ref:
      assert snobj.model.parameters[p] == pytest.approx(ref[p], abs=1e-6)
   assert num.isfinite(snobj.model.rchisquare)


def test_non_finite_model_at_used_points_raises(snobj):
   snobj.replot = False
   snobj.choose_model('max_model', stype='st')
   _inject(snobj.model, 'V', 3, num.nan, masked=False)
   with pytest.raises(RuntimeError, match="not finite"):
      snobj.fit(bands=['B', 'V'], dokcorr=False)


def test_dof_counts_only_data_used_in_fit(snobj):
   snobj.replot = False
   snobj.choose_model('max_model', stype='st')
   snobj.data['B'].mask[:5] = False
   snobj.fit(bands=['B', 'V'], dokcorr=False)
   m = snobj.model
   used = 0
   for b in ['B', 'V']:
      mask = m(b, snobj.data[b].MJD)[2]
      used += num.sum(num.asarray(mask, dtype=bool) & snobj.data[b].mask)
   assert used < len(snobj.data['B'].MJD) + len(snobj.data['V'].MJD)
   assert m.dof == used - len(m._free)
   assert m.rchisquare == pytest.approx(m.chisquare/m.dof)


def test_user_covariance_gives_exact_chi2(snobj):
   """With a full covariance matrix, chi2 must be d^T C^-1 d."""
   snobj.replot = False
   snobj.choose_model('max_model', stype='st')
   B = snobj.data['B']
   # photometric errors plus a correlated (zero-point like) term, in mags
   B.covar = num.diag(B.e_mag**2) + 0.02**2
   snobj.fit(bands=['B', 'V'], dokcorr=False)
   m = snobj.model
   mod, err, mask = m('B', B.MJD)
   used = num.asarray(mask, dtype=bool) & B.mask
   f = num.power(10, -0.4*(mod - B.filter.zp))
   C = B.get_covar(flux=1) + num.diag((f*err/1.0857)**2)
   C = C[used][:,used]
   d = (f - B.flux)[used]
   r = m._wrap_model([m.parameters[p] for p in m._free], ['B'],
                     {'B': B.get_covar(flux=1)})
   assert num.sum(r**2) == pytest.approx(d @ num.linalg.solve(C, d), rel=1e-9)


def _count_kcorr_and_nudge_Tmax(snobj, nudge_calls):
   """Count k-correction passes; after the given model.fit() calls
   (1-based) shift Tmax by 0.1 d times the call number, to emulate a fit
   that keeps moving after the k-corrections were computed."""
   state = {'fits': 0, 'kcorrs': []}
   kc = snobj.kcorr
   def kcorr(*a, **k):
      state['kcorrs'].append(snobj.Tmax)
      return kc(*a, **k)
   snobj.kcorr = kcorr
   fit = snobj.model.fit
   def mfit(*a, **k):
      fit(*a, **k)
      state['fits'] += 1
      if state['fits'] in nudge_calls:
         snobj.model.parameters['Tmax'] += 0.1*state['fits']
   snobj.model.fit = mfit
   return state


@pytest.mark.parametrize("kcorr_iter,nudges,n_kcorr", [
   (3, {3}, 5),          # moves on fit 3, back on 4 (damped), settles: 3 passes
   (0, {3, 4}, 2),       # old behaviour: k-corrections computed once
   (3, set(), 2),        # already converged: no extra pass
])
def test_fit_iterates_kcorrs_until_converged(snobj, kcorr_iter, nudges, n_kcorr):
   import warnings
   snobj.replot = False
   snobj.choose_model('max_model', stype='st')
   state = _count_kcorr_and_nudge_Tmax(snobj, nudges)
   with warnings.catch_warnings():
      warnings.simplefilter('error')    # no non-convergence warning
      snobj.fit(bands=['B', 'V'], kcorr_iter=kcorr_iter)
   assert len(state['kcorrs']) == n_kcorr
   if kcorr_iter:
      assert abs(snobj.Tmax - state['kcorrs'][-1]) < 0.05


def test_fit_warns_if_kcorrs_do_not_converge(snobj):
   snobj.replot = False
   snobj.choose_model('max_model', stype='st')
   state = _count_kcorr_and_nudge_Tmax(snobj, set(range(3, 100)))
   with pytest.warns(UserWarning, match="did not converge"):
      snobj.fit(bands=['B', 'V'], kcorr_iter=2)
   assert len(state['kcorrs']) == 4


@pytest.mark.parametrize("inflate", [1.0, 3.0])     # chi2_nu > 1 and < 1
def test_error_scaling_modes(snobj, inflate):
   snobj.replot = False
   for b in ['B', 'V']:
      snobj.data[b].e_mag = snobj.data[b].e_mag*inflate
   errs = {}
   for mode in ['none', 'always', 'up']:
      snobj.choose_model('max_model', stype='st')
      snobj.fit(bands=['B', 'V'], dokcorr=False, error_scaling=mode)
      errs[mode] = dict(snobj.model.errors)
      r = snobj.model.rchisquare
   assert (r > 1) == (inflate == 1.0)
   for p in ['st', 'Bmax', 'Vmax']:
      assert errs['always'][p] == pytest.approx(errs['none'][p]*num.sqrt(r), rel=1e-6)
      assert errs['up'][p] == pytest.approx(errs['none'][p]*num.sqrt(max(1, r)), rel=1e-6)


def test_error_scaling_rejects_unknown_mode(snobj):
   snobj.replot = False
   snobj.choose_model('max_model', stype='st')
   with pytest.raises(ValueError, match="error_scaling"):
      snobj.fit(bands=['B', 'V'], dokcorr=False, error_scaling='down')


def test_fit_damps_oscillating_kcorr_iterations(snobj):
   """A fit whose Tmax overshoots the k-correction Tmax by -1.5x never
   converges undamped (0.5, -0.75, 1.1, ...); halving converges."""
   import warnings
   snobj.replot = False
   snobj.choose_model('max_model', stype='st')
   state = {'fits': 0, 'kTmax': []}
   kc = snobj.kcorr
   def kcorr(*a, **k):
      state['kTmax'].append(snobj.Tmax)
      return kc(*a, **k)
   snobj.kcorr = kcorr
   fit = snobj.model.fit
   def mfit(*a, **k):
      fit(*a, **k)
      state['fits'] += 1
      if state['fits'] == 2:
         state['T0'] = snobj.Tmax
      elif state['fits'] == 3:
         snobj.model.parameters['Tmax'] = state['T0'] + 0.5
      elif state['fits'] > 3:
         snobj.model.parameters['Tmax'] = state['T0'] - 1.5*(state['kTmax'][-1] - state['T0'])
   snobj.model.fit = mfit
   with warnings.catch_warnings():
      warnings.simplefilter('error')
      snobj.fit(bands=['B', 'V'], kcorr_iter=5)
   assert abs(snobj.Tmax - state['T0']) < 0.05
   assert abs(snobj.Tmax - state['kTmax'][-1]) < 0.05
