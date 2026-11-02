'''Tests for the Milky-Way E(B-V) from local SFD maps and for the dust laws
computed with the ``extinction`` package.'''
import os
import numpy as np
import pytest
import extinction
from snpy.utils import sfd_dust, deredden

# SN2006ax
RA, DEC = 171.014417, -12.291444


# The un-patched function (conftest pins sfd_dust.get_ebv to 0.041); test
# modules are imported before the fixtures are applied.
_get_ebv = sfd_dust.get_ebv


@pytest.fixture
def mapdir():
   d = os.environ.get('SFD_DIR')
   if not d or not os.path.isfile(os.path.join(d, 'SFD_dust_4096_ngp.fits')):
      pytest.skip("SFD maps not available (set SFD_DIR)")
   return d


def test_sfd_value(mapdir):
   get_ebv = _get_ebv
   sf11 = get_ebv(RA, DEC, mapdir=mapdir)
   sfd98 = get_ebv(RA, DEC, calibration='SFD98', mapdir=mapdir)
   # IRSA's 5-arcmin mean is 0.041; the map value at the position is 0.0432
   assert sf11 == pytest.approx(0.0432, abs=5e-4)
   assert sf11 / sfd98 == pytest.approx(0.86)


def test_missing_maps_raise(tmp_path, monkeypatch):
   monkeypatch.delenv('SFD_DIR', raising=False)
   monkeypatch.setattr(sfd_dust, 'AUTO_DOWNLOAD', False)
   monkeypatch.setattr(sfd_dust, 'package_dir', lambda: str(tmp_path / 'pkg'))
   monkeypatch.setattr(sfd_dust, 'cache_dir', lambda: str(tmp_path / 'cache'))
   get_ebv = _get_ebv
   with pytest.raises(sfd_dust.SFDMapError):
      get_ebv(RA, DEC)
   # an explicit directory without the maps is an error, never a download
   monkeypatch.setattr(sfd_dust, 'AUTO_DOWNLOAD', True)
   with pytest.raises(sfd_dust.SFDMapError):
      get_ebv(RA, DEC, mapdir=str(tmp_path))


def _fake_server(tmp_path, monkeypatch):
   src = tmp_path / 'server'
   src.mkdir(exist_ok=True)
   md5s = {}
   for f in sfd_dust.FILES:
      (src / f).write_bytes(f.encode() * 1000)
      md5s[f] = sfd_dust._md5(str(src / f))
   monkeypatch.setattr(sfd_dust, 'BASE_URL', src.as_uri() + '/')
   monkeypatch.setattr(sfd_dust, 'FILES', md5s)
   monkeypatch.delenv('SFD_DIR', raising=False)
   return md5s


def test_auto_download(tmp_path, monkeypatch):
   '''Missing maps are downloaded into the installation directory, or the
   user cache when that is not writable, and then found there.'''
   _fake_server(tmp_path, monkeypatch)
   pkg, cache = tmp_path / 'pkg', tmp_path / 'cache'
   monkeypatch.setattr(sfd_dust, 'package_dir', lambda: str(pkg))
   monkeypatch.setattr(sfd_dust, 'cache_dir', lambda: str(cache))
   assert sfd_dust.get_mapdir() == str(pkg)
   assert sorted(p.name for p in pkg.iterdir()) == sorted(sfd_dust.FILES)
   assert all(p.stat().st_mode & 0o044 for p in pkg.iterdir())
   # second call finds them without downloading
   monkeypatch.setattr(sfd_dust, 'download', lambda *a, **k: 1/0)
   assert sfd_dust.get_mapdir() == str(pkg)

   monkeypatch.undo()
   _fake_server(tmp_path, monkeypatch)
   ro = tmp_path / 'readonly'
   monkeypatch.setattr(sfd_dust, 'package_dir', lambda: str(ro))
   monkeypatch.setattr(sfd_dust, 'cache_dir', lambda: str(cache))
   monkeypatch.setattr(sfd_dust, '_writable', lambda d: d == str(cache))
   assert sfd_dust.get_mapdir() == str(cache)
   assert not ro.exists()


def test_download_bad_checksum(tmp_path, monkeypatch):
   _fake_server(tmp_path, monkeypatch)
   monkeypatch.setattr(sfd_dust, 'FILES', {f: '0'*32 for f in sfd_dust.FILES})
   with pytest.raises(sfd_dust.SFDMapError, match='Checksum'):
      sfd_dust.download(str(tmp_path / 'out'), verbose=False)
   # no partial file is left behind
   assert list((tmp_path / 'out').iterdir()) == []


def test_missing_maps_sn(datadir, monkeypatch):
   '''Without maps, EBVgal is None (with a warning) and fit() refuses.'''
   import snpy

   def _fail(*args, **kw):
      raise sfd_dust.SFDMapError("no maps")
   monkeypatch.setattr(sfd_dust, 'get_ebv', _fail)
   with pytest.warns(UserWarning, match='E\\(B-V\\)_gal not computed'):
      s = snpy.get_sn(os.path.join(datadir, 'SN2006ax.txt'))
   assert s.EBVgal is None
   s.replot = 0
   s.choose_model('EBV_model2', stype='st')
   with pytest.raises(ValueError, match='EBVgal'):
      s.fit(['B', 'V'])
   s.EBVgal = 0.041
   s.fit(['B', 'V'], dokcorr=False)


def test_ccm_matches_extinction():
   w = np.linspace(1000., 30000., 5000)
   for rv in [1.5, 3.1]:
      a, b = deredden.ccm(w)
      assert np.allclose(rv*a + b, extinction.odonnell94(w, rv, rv))
      a, b = deredden.ccm(w, strict_ccm=1)
      assert np.allclose(rv*a + b, extinction.ccm89(w, rv, rv))
   # CCM89 eq. 4: mid-UV at x = 5/micron, R_V = 3.1
   x = 5.
   a = 1.752 - 0.316*x - 0.104/((x - 4.67)**2 + 0.341)
   b = -3.090 + 1.825*x + 1.206/((x - 4.62)**2 + 0.263)
   a2, b2 = deredden.ccm(np.array([2000.]))
   assert a2[0] == pytest.approx(a) and b2[0] == pytest.approx(b)
   # shape handling and zero outside the defined range
   assert np.ndim(deredden.ccm(5500.)[0]) == 0
   assert deredden.ccm(np.array([5e4]))[0][0] == 0


def test_fm_matches_extinction():
   w = np.linspace(1000., 30000., 5000)
   assert np.allclose(deredden.fm(w, 2.5), extinction.fitzpatrick99(w, 2.5, 2.5))
