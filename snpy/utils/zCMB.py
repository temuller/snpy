'''
Conversions of heliocentric redshifts to the CMB and Local Group frames,
computed locally from the coordinates (z_cmb_NED is the old NED query, kept
for reference).'''


import urllib.request as urllib
import re
from astropy.coordinates import SkyCoord
from math import sin,cos,pi,sqrt
import numpy as np
url_temp = "http://ned.ipac.caltech.edu/cgi-bin/velc?lon=%.6fd&in_csys=Equatorial&lat=%.6fd&in_equinox=J2000.0&vel=%.3f&vfrom=Heliocentric&vto=3K&alon=&a_csys=Equatorial&alat=&a_equinox=J2000.0&avel=0.0"
pat = re.compile(r'([0-9]+(\.[0-9]*)?)')

# CMB dipole: Planck 2018 (Planck Collaboration 2020, A&A 641, A1)
CMB_l = 264.021     # galactic longitude (deg)
CMB_b = 48.253      # galactic latitude (deg)
CMB_V = 369.82      # Sun's velocity relative to the CMB (km/s)
C_KMS = 299792.458  # speed of light (km/s)

def z_cmb(z_hel, ra, dec):
   '''heliocentric to CMB redshift conversion.

   Redshifts compound: (1 + z_cmb) = (1 + z_hel)/(1 + z_sun), where
   z_sun = -V cos(theta)/c is the redshift from the Sun's motion relative to
   the CMB along the line of sight (theta: angle to the dipole direction).

      z_hel:  heliocentric redshift of object (float or array)
      ra,dec: coordinates in degrees (ICRS).
   '''
   cmb_dir = SkyCoord(l=CMB_l, b=CMB_b, frame='galactic', unit='deg')
   obj_dir = SkyCoord(ra, dec, frame='icrs', unit='deg')
   cos_sep = np.cos(cmb_dir.separation(obj_dir).radian)
   z_sun = -CMB_V*cos_sep/C_KMS
   return (1 + np.asarray(z_hel, dtype=float))/(1 + z_sun) - 1

def v_LG(z_hel, ra, dec):
   '''heliocentric to LG velocity conversion.
      z_hel:  heliocentric redshift of object
      ra,dec: coorinates in degrees.
   '''
   v_hel = z_hel*3e5
   obj_dir = SkyCoord(ra, dec, frame='icrs', unit='deg')
   l = obj_dir.galactic.l.to('rad').value
   b = obj_dir.galactic.b.to('rad').value
   V_LG = v_hel - 79*cos(l)*cos(b) + 296*sin(l)*cos(b)-36*sin(b)
   return V_LG

def v_infall(Vo, ra, dec, Vfid, Va, ra_a, dec_a, gamma=2):
   '''Compute an infall velocity due to an attractor.  See Mould+2000
      ra,dec:  coordinates of object in deg
      Vo:      LG-frame velocity of object
      Vfid:    amplitude of attractor at LG
      Va:      distance to attractor in km/s
      la,lb:   galactic coordinates of attractor
      gamma:   slope of attractor's density profile
   '''
   obj_dir = SkyCoord(ra, dec, frame='icrs', unit='deg')
   att_dir = SkyCoord(ra_a, dec_a, frame='icrs', unit='deg')
   theta = att_dir.separation(obj_dir).to('rad').value
   # projected distance
   roa = sqrt(Vo**2 + Va**2 - 2*Va*Vo*cos(theta))
   Vinf = Vfid*cos(theta) + Vfid*((Vo - Va*cos(theta))/roa)* pow(roa/Va,1-gamma)
   return Vinf

def z_cmb_NED(z_hel, ra, dec):
   v_hel = z_hel*3e5
   u = urllib.urlopen(url_temp % (ra, dec, v_hel))
   lines = u.readlines()
   vcmb = None
   for line in lines:
      if line.find('Output:') >= 0:
         res = pat.search(line)
         if res is not None:
            vcmb = float(res.groups()[0])
   if vcmb is None:
      raise ValueError("Could not parse output from NED")
   return vcmb/3e5
