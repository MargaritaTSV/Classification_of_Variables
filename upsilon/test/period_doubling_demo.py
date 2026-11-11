"""
Demo: P-vs-2P period-doubling check for eclipsing binaries.

Builds a synthetic eclipsing-binary light curve whose two eclipses differ
in depth -- the exact scenario that makes a single-harmonic Lomb-Scargle
lock onto half the true period (true P=0.4 d would alias into the delta
Scuti zone, P<=0.25 d). Runs the UPSILoN feature extractor with
`save_diagnostics=True` so a plot and a JSON report of the P-vs-2P check
are written to `diagnostics/`, then prints the before/after period values.

Also runs the same pipeline on the bundled real EROS sample light curve as
a sanity check that ordinary (non-EB) stars are not spuriously doubled.
"""

import numpy as np

from upsilon.extract_features.extract_features import ExtractFeatures
from upsilon.datasets.base import load_EROS_lc


def make_synthetic_eb(period=0.4, n_points=600, baseline_days=120.,
                       depth1=0.6, depth2=0.25, width=0.06, noise=0.01,
                       seed=0):
    """
    Generate a synthetic eclipsing-binary light curve.

    Two sharp, symmetric eclipses per true cycle, with unequal depths
    (primary and secondary minimum), on an irregular (survey-like) time
    sampling.

    Parameters
    ----------
    period : float
        True orbital period, in days.
    n_points : int
        Number of observations.
    baseline_days : float
        Total time span of the light curve, in days.
    depth1, depth2 : float
        Depths of the primary and secondary eclipse, in magnitudes.
    width : float
        Eclipse width, as a fraction of the period.
    noise : float
        Gaussian photometric noise, in magnitudes.
    seed : int
        Random seed.

    Returns
    -------
    date : numpy.ndarray
    mag : numpy.ndarray
    err : numpy.ndarray
    """
    rng = np.random.RandomState(seed)

    date = np.sort(rng.uniform(0., baseline_days, n_points))
    phase = (date % period) / period

    mag = np.full(n_points, 15.0)

    def eclipse(phase, center, depth, width):
        dist = np.abs(phase - center)
        dist = np.minimum(dist, 1. - dist)
        return depth * np.exp(-0.5 * (dist / width) ** 2)

    mag += eclipse(phase, 0.0, depth1, width)
    mag += eclipse(phase, 0.5, depth2, width)

    err = np.ones(n_points) * noise
    mag += rng.normal(0., noise, n_points)

    return date, mag, err


def run():
    print('=== Synthetic eclipsing binary ===')
    true_period = 0.4
    date, mag, err = make_synthetic_eb(period=true_period)

    features = ExtractFeatures(date, mag, err, save_diagnostics=True,
                                diagnostics_dir='diagnostics',
                                star_id='synthetic_EB')
    features.run()

    print('True period:            %.6f d' % true_period)
    print('Raw LS period (before): %.6f d' % features.ls_raw_period)
    print('Final period (after):   %.6f d' % features.period)
    print('BIC(P):                 %.3f' % features.bic_period)
    print('BIC(2P):                %.3f' % features.bic_double_period)
    print('Doubled:                %s' % features.period_doubled)
    print()

    print('=== Bundled real light curve (lm0010n22323) ===')
    date, mag, err = load_EROS_lc()
    index = mag < 99.999
    date, mag, err = date[index], mag[index], err[index]

    features = ExtractFeatures(date, mag, err, save_diagnostics=True,
                                diagnostics_dir='diagnostics',
                                star_id='lm0010n22323')
    features.run()

    print('Raw LS period (before): %.6f d' % features.ls_raw_period)
    print('Final period (after):   %.6f d' % features.period)
    print('BIC(P):                 %.3f' % features.bic_period)
    print('BIC(2P):                %.3f' % features.bic_double_period)
    print('Doubled:                %s' % features.period_doubled)
    print()

    print('Diagnostics written to ./diagnostics/'
          '{synthetic_EB,lm0010n22323}_period_check.{png,json}')


if __name__ == '__main__':
    run()
