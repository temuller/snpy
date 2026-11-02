import pytest
import os
from snpy import get_sn
from snpy.utils import fit1dcurve
from snpy.version import __version__

@pytest.fixture
def snobj(datadir):
   import snpy
   snobj = snpy.get_sn(os.path.join(datadir, 'SN2006ax.txt'))
   for f in snobj.data:
      snobj.data[f].template(method='spline2')
   return snobj

# Test computing of bolometric light-curve

def test_bolo_direct(snobj):
   x,y,f,lims = snobj.bolometric(['u','B','V','r','i','Y','J','H'], 
      method='direct', interpolate='spline', SED='H3', verbose=True,
      EBVhost=0, Rv=3.1)
   checksum = sum(y)*1e-44
   assert checksum == pytest.approx(2.239, abs=0.002)

def test_bolo_direct_model(snobj):
   
   snobj.choose_model('color_model', stype='st')
   snobj.fit(dokcorr=False, Rv=3.1)
   x,y,f,lims = snobj.bolometric(['u','B','V','r','i','Y','J','H'], 
      method='direct', interpolate='model', SED='H3')
   checksum = sum(y)*1.e-44
   assert checksum == pytest.approx(2.566, abs=0.002)

def test_bolo_SED(snobj):
   from snpy import fset
   l0 = fset['u'].wave.min()
   l1 = fset['H'].wave.max()
   x,y,f,lims = snobj.bolometric(['u','B','V','r','i','Y','J','H'], 
      method='SED', interpolate='spline', refband='B', lam1=l0,
      lam2=l1, SED='H3', EBVhost=0, Rv=3.1)
   checksum = sum(y)*1.e-44
   assert checksum == pytest.approx(2.709, abs=0.002)


def test_bolo_SED_tmin_tmax(snobj):
   """tmin/tmax must both be applied (AND) and keep mags/masks aligned.

   Regression test: tmax used to overwrite tmin, and the per-filter mags/
   masks arrays were not re-sliced to match the restricted epochs.
   """
   import numpy as num
   from snpy import fset
   l0 = fset['u'].wave.min()
   l1 = fset['H'].wave.max()
   bands = ['u','B','V','r','i','Y','J','H']
   tmin, tmax = -10, 30
   r = snobj.bolometric(bands,
      method='SED', interpolate='spline', refband='B', lam1=l0,
      lam2=l1, SED='H3', EBVhost=0, Rv=3.1, tmin=tmin, tmax=tmax,
      extra_output=True)
   epochs = num.asarray(r[0])
   extra = r[4]
   # both bounds applied
   assert epochs.min() > tmin
   assert epochs.max() < tmax
   # epochs, flux and the per-filter arrays all stay the same length
   assert len(epochs) == len(r[1]) == len(r[2])
   assert extra['mags'].shape == (len(epochs), len(bands))
   assert extra['masks'].shape == extra['mags'].shape

