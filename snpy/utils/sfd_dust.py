'''Milky-Way E(B-V) from local copies of the Schlegel, Finkbeiner & Davis
(1998) dust maps, read with the ``sfdmap2`` package.

The maps are not shipped with SNooPy. Download them once, e.g. from
https://github.com/kbarbary/sfddata (``SFD_dust_4096_ngp.fits`` and
``SFD_dust_4096_sgp.fits``), and tell SNooPy where they are, either with the
``SFD_DIR`` environment variable or by setting ``sfd_dust.SFD_DIR``:

   >>> from snpy.utils import sfd_dust
   >>> sfd_dust.SFD_DIR = '/path/to/sfddata'
'''
import os

#: Directory holding the SFD map FITS files. If None, the ``SFD_DIR``
#: environment variable is used.
SFD_DIR = None

# Schlafly & Finkbeiner (2011) recalibration of the SFD98 E(B-V)
SCALINGS = {'SF11': 0.86, 'SFD98': 1.0}

_maps = {}


class SFDMapError(RuntimeError):
   pass


def get_mapdir(mapdir=None):
   '''The SFD map directory: mapdir, else SFD_DIR, else $SFD_DIR.'''
   if mapdir is None:
      mapdir = SFD_DIR
   if mapdir is None:
      mapdir = os.environ.get('SFD_DIR')
   if not mapdir:
      raise SFDMapError("The location of the SFD dust maps is not set. Set "
            "the SFD_DIR environment variable (or snpy.utils.sfd_dust.SFD_DIR) "
            "to the directory holding SFD_dust_4096_ngp.fits and "
            "SFD_dust_4096_sgp.fits, or set the SN's EBVgal by hand.")
   mapdir = os.path.expanduser(mapdir)
   for pole in ['ngp', 'sgp']:
      f = os.path.join(mapdir, 'SFD_dust_4096_%s.fits' % pole)
      if not os.path.isfile(f):
         raise SFDMapError("SFD dust map %s not found. Check SFD_DIR, or set "
               "the SN's EBVgal by hand." % f)
   return mapdir


def get_ebv(ra, dec, calibration='SF11', mapdir=None):
   '''Milky-Way E(B-V) at (ra, dec) in degrees (ICRS/J2000).

   Args:
      ra, dec (float or array): coordinates in degrees.
      calibration (str): 'SF11' (Schlafly & Finkbeiner 2011: 0.86 x SFD98)
                         or 'SFD98'.
      mapdir (str): directory of the map files (default: see get_mapdir).

   Returns:
      float or array: E(B-V), interpolated in the map.

   Raises:
      SFDMapError: if the maps cannot be found.
   '''
   if calibration not in SCALINGS:
      raise ValueError("calibration must be one of %s" % list(SCALINGS))
   from sfdmap2 import sfdmap
   mapdir = get_mapdir(mapdir)
   key = (mapdir, calibration)
   if key not in _maps:
      _maps[key] = sfdmap.SFDMap(mapdir, scaling=SCALINGS[calibration])
   return _maps[key].ebv(ra, dec, frame='icrs', unit='degree')
