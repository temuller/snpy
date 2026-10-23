# EBV_model2 calibration for the generation-3 templates

`EBV_model2` takes its peak absolute magnitudes, their dependence on `st`, the
host-galaxy R_V and the intrinsic scatter from `snpy/st_calibration2.dat`.
That file was added in May 2012 (by C. Burns), nine years before the
generation-3 light-curve templates (August 2021), so fitting `EBV_model2` with
`gen=3.1` mixes a calibration made with gen-1/2 templates with gen-3 light
curves. The scripts here rebuild the calibration so it can be redone for any
template generation and SED.

## Provenance of `st_calibration2.dat`

Nothing in the repository records how the file was made. What could be
established:

- It follows the "color excess" calibration of Burns et al. 2011 (AJ 141, 19),
  extended from Δm15 to `st` with a quadratic term.
- Its comment lines define the six variants (the `EBV_model2` docstring
  describes them incorrectly): 0/1/2 = with u-band, and st > 0.5 /
  (B−V)max < 0.3 / all objects; 3/4/5 = the same without u-band.
- The sample was presumably the CSP-I SNe Ia of data releases 1 and 2
  (Contreras et al. 2010; Stritzinger et al. 2011) in the Hubble flow.
- The R_V values (1.52–1.75) do not match Burns+2011 (1.40–2.64) or Burns+2014.

## Method (`calibrate.py`)

For every SN *j* and rest band *X*, using the peak magnitudes from `max_model`:

    m_Xj = a_X + b_X (st_j − 1) + c_X (st_j − 1)² + R_X(R_V) E_j + μ_j

- One host E_j ≥ 0 per SN. The Folatelli et al. 2010 low-reddening SNe are
  anchored at E_j = 0 (the Burns+2011 "blue prior"). This is essential:
  without it the reddening absorbs the faint, red fast decliners and the
  quadratic terms come out ~8σ off.
- R_X is computed as `EBV_model2` does: `kcorr.R_obs` of the SED at maximum
  (with the SN's colour stretch for H3+L), E(B−V) = 0.01, CCM+O'Donnell law.
- μ_j from the CMB-frame redshift with H0 = 72, Ω_M = 0.28 (Burns+2011), so
  the DM zero point of `EBV_model2` is unchanged.
- Intrinsic scatter per band, no separate peculiar-velocity term (this is
  what reproduces the file's scatters; `--sig-v` adds one).
- The `st` errors and their covariance with the peaks are propagated through
  dM/dst. E_j is solved analytically; the global parameters maximize the
  exact likelihood. Errors are half the 16–84% range of a bootstrap over the
  SNe: resamples that miss most fast decliners leave the quadratic term
  unconstrained, and those heavy tails would dominate a standard deviation.

## Validation

On the 2012 sample (55 SNe for variant 0) with gen-2 templates, the H3 SED
and CCM, the method reproduces calibration 0 of `st_calibration2.dat` within
2σ in every parameter: stretch terms within 0–0.4σ in u/B/V/g/r and
0.5–1.6σ in i/Y/J/H, and the intrinsic scatters closely. Two differences
remain: the zero points are ~0.05 mag fainter in every band (0.5–1.8σ each;
likely the DR3 recalibration of the photometry, the distance convention, or
the 10 SNe of the 2012 sample that are not in the SAINTS data), and
R_V = 1.82 ± 0.15 vs 1.52 ± 0.08 (1.8σ).

## Results for gen=3.1 (H3+L SED, CCM)

`results/st_calibration_gen3.1_2012.dat` and
`results/st_calibration_gen3.1_extended.dat` hold the six variants for each
sample, in the format of `st_calibration2.dat`.

**2012 sample** (53–55 SNe): consistent with `st_calibration2.dat`. Every
parameter agrees within ~2σ; the stretch terms within 0–1.7σ. The zero points
are 0.01–0.10 mag fainter (+0.06 typical), the same offset as the gen-2
re-derivation, so it is not caused by the templates. R_V = 1.77 ± 0.20 for
variant 0 (1.77–2.13 across variants). Switching from gen-2 to gen-3.1
templates moves the calibration by only ~0.02 mag, so `st_calibration2.dat`
remains adequate for `gen=3.1` within its uncertainties, and it was kept.

**Extended sample** (all CSP-I + CSP-II, 223–241 SNe): not usable as is. The
zero points are ~0.1 mag brighter than the file and the residuals trend with
redshift (mean B residual +0.04 at z < 0.02, −0.03 at z > 0.05), as expected
from Malmquist (selection) bias in the more distant CSP-II SNe. R_V rises to
2.3 (3.1–3.5 for the (B−V) < 0.3 variants). It would need a selection-bias
correction or a redshift-limited sample.

Open question: the ~0.05 mag zero-point offset of the 2012-sample
re-derivations with respect to `st_calibration2.dat` (DR3 vs 2012 photometry,
distance convention, or the 10 missing SNe).

## Running it

The light curves and the sample table come from the SAINTS compilation
(`../../saints/saints_photometry`).

    export PYTHONPATH=/path/to/snpy
    python fit_peaks.py --sample 2012 --gen 3.1 --kversion H3+L --out results/fits.json
    for v in 0 1 2 3 4 5; do
       python calibrate.py results/fits.json --variant $v --nboot 100 --out results/cal$v.json
    done
    python write_calfile.py results/cal?.json --out st_calibration3.dat
    python compare.py results/cal?.json        # against st_calibration2.dat

`--sample extended` uses every CSP-I and CSP-II SN Ia in the compilation.
Peculiar SNe (pec, Iax, super-Chandrasekhar, 06bt-, 02es-, 06gz-like) are
excluded using the `Type` column of the SAINTS sample table.
