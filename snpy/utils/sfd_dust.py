'''Milky-Way E(B-V) from local copies of the Schlegel, Finkbeiner & Davis
(1998) dust maps, read with the ``sfdmap2`` package.

The maps (two FITS files, 64 MB each) are looked for, in order, in:

1. the ``mapdir`` argument of :func:`get_ebv`;
2. ``snpy.utils.sfd_dust.SFD_DIR``;
3. the ``SFD_DIR`` environment variable;
4. ``sfddata/`` in the SNooPy installation directory;
5. ``snpy/sfddata`` in the user cache (``$XDG_CACHE_HOME`` or ``~/.cache``).

If they are not found and ``AUTO_DOWNLOAD`` is True (the default), they are
downloaded from https://github.com/kbarbary/sfddata the first time they are
needed, into the SNooPy installation directory or, if that is not writable,
into the user cache. :func:`download` does this explicitly.
'''
import os
import hashlib
import shutil
import tempfile
import urllib.request

#: Directory holding the SFD map FITS files. If None, see the module docs.
SFD_DIR = None

#: Download the maps automatically when they are not found.
AUTO_DOWNLOAD = True

BASE_URL = 'https://github.com/kbarbary/sfddata/raw/master/'
FILES = {'SFD_dust_4096_ngp.fits': '6f3a89528bba7d1b7bd55838da86b639',
         'SFD_dust_4096_sgp.fits': '60f7c43d51bccaff82961de3e9f9e789'}

# Schlafly & Finkbeiner (2011) recalibration of the SFD98 E(B-V)
SCALINGS = {'SF11': 0.86, 'SFD98': 1.0}

_maps = {}


class SFDMapError(RuntimeError):
   pass


def package_dir():
   '''sfddata/ in the SNooPy installation directory.'''
   return os.path.join(os.path.dirname(os.path.dirname(
      os.path.abspath(__file__))), 'sfddata')


def cache_dir():
   '''snpy/sfddata in the user cache directory.'''
   base = os.environ.get('XDG_CACHE_HOME') or \
         os.path.join(os.path.expanduser('~'), '.cache')
   return os.path.join(base, 'snpy', 'sfddata')


def _has_maps(d):
   return all(os.path.isfile(os.path.join(d, f)) for f in FILES)


def _writable(d):
   '''True if d can be created and written to.'''
   try:
      os.makedirs(d, exist_ok=True)
      with tempfile.NamedTemporaryFile(dir=d):
         pass
      return True
   except OSError:
      return False


def _md5(path):
   h = hashlib.md5()
   with open(path, 'rb') as f:
      for chunk in iter(lambda: f.read(1 << 20), b''):
         h.update(chunk)
   return h.hexdigest()


def download(mapdir=None, verbose=True):
   '''Download the SFD maps into mapdir (default: the SNooPy installation
   directory if writable, else the user cache) and return that directory.
   Files already there with the right checksum are kept.'''
   if mapdir is None:
      for d in [package_dir(), cache_dir()]:
         if _writable(d):
            mapdir = d
            break
      else:
         raise SFDMapError("Cannot write the SFD dust maps to %s or %s. "
               "Download them yourself from %s and set SFD_DIR." %
               (package_dir(), cache_dir(), BASE_URL))
   mapdir = os.path.expanduser(mapdir)
   os.makedirs(mapdir, exist_ok=True)
   for fname, md5 in FILES.items():
      dest = os.path.join(mapdir, fname)
      if os.path.isfile(dest) and (md5 is None or _md5(dest) == md5):
         continue
      if verbose:
         print("Downloading the SFD dust map %s (64 MB) to %s ..." %
               (fname, mapdir))
      # Download to a temporary file and move it into place, so that an
      # interrupted or concurrent download never leaves a partial map.
      fd, tmp = tempfile.mkstemp(dir=mapdir, suffix='.part')
      try:
         with os.fdopen(fd, 'wb') as out, \
               urllib.request.urlopen(BASE_URL + fname, timeout=60) as u:
            shutil.copyfileobj(u, out, 1 << 20)
         if md5 is not None and _md5(tmp) != md5:
            raise SFDMapError("Checksum mismatch for the downloaded %s" % fname)
         os.chmod(tmp, 0o644)    # mkstemp makes it private
         os.replace(tmp, dest)
      except Exception as e:
         if os.path.exists(tmp):
            os.remove(tmp)
         if isinstance(e, SFDMapError):
            raise
         raise SFDMapError("Could not download the SFD dust map %s: %s. "
               "Download it yourself from %s and set SFD_DIR." %
               (fname, e, BASE_URL)) from e
   return mapdir


def get_mapdir(mapdir=None):
   '''The SFD map directory (see the module docs for the search order),
   downloading the maps if needed and AUTO_DOWNLOAD is True.'''
   if mapdir is None:
      mapdir = SFD_DIR
   if mapdir is None:
      mapdir = os.environ.get('SFD_DIR') or None
   if mapdir is not None:
      # An explicit location must hold the maps
      mapdir = os.path.expanduser(mapdir)
      if not _has_maps(mapdir):
         raise SFDMapError("SFD dust maps (%s) not found in %s. Check "
               "SFD_DIR, or set the SN's EBVgal by hand." %
               (' and '.join(FILES), mapdir))
      return mapdir
   for d in [package_dir(), cache_dir()]:
      if _has_maps(d):
         return d
   if not AUTO_DOWNLOAD:
      raise SFDMapError("SFD dust maps not found. Run "
            "snpy.utils.sfd_dust.download(), set SFD_DIR to a directory "
            "holding them, or set the SN's EBVgal by hand.")
   return download()


def get_ebv(ra, dec, calibration='SF11', mapdir=None):
   '''Milky-Way E(B-V) at (ra, dec) in degrees (ICRS/J2000).

   Args:
      ra, dec (float or array): coordinates in degrees.
      calibration (str): 'SF11' (Schlafly & Finkbeiner 2011: 0.86 x SFD98)
                         or 'SFD98'.
      mapdir (str): directory of the map files (default: see module docs).

   Returns:
      float or array: E(B-V), interpolated in the map.

   Raises:
      SFDMapError: if the maps cannot be found or downloaded.
   '''
   if calibration not in SCALINGS:
      raise ValueError("calibration must be one of %s" % list(SCALINGS))
   from sfdmap2 import sfdmap
   mapdir = get_mapdir(mapdir)
   key = (mapdir, calibration)
   if key not in _maps:
      _maps[key] = sfdmap.SFDMap(mapdir, scaling=SCALINGS[calibration])
   return _maps[key].ebv(ra, dec, frame='icrs', unit='degree')
