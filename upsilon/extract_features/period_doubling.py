"""
P vs 2P period-doubling check for eclipsing binaries.

A single-harmonic Lomb-Scargle fit (one sine + one cosine) locks onto half
the true period whenever an eclipsing binary's two eclipses are similar,
since one sinusoid per cycle fits the data better at twice the true
frequency. This module decides whether a period found that way should be
doubled, by comparing BIC-penalized multi-harmonic Fourier fits at the
candidate period P and at 2P.

The 2P fit is built with twice as many harmonics as the P fit, so its basis
is a strict superset of the P basis: its even harmonics reproduce the P
model exactly, and its odd harmonics are the only terms that can express two
unequal eclipses per true cycle. Comparing raw residuals would then always
favor 2P (more basis functions can only fit the data at least as well);
BIC's k*ln(n) penalty is what makes the extra odd-harmonic terms earn their
keep only when they truly reduce scatter, not just fit noise.
"""

import numpy as np
from scipy.optimize import leastsq


def fourier_series(pars, phase, order):
    """
    Evaluate a Fourier series at given phases.

    Parameters
    ----------
    pars : array_like
        Fourier coefficients: [mean, a1, b1, a2, b2, ...].
    phase : array_like
        Phases in [0, 1).
    order : int
        Number of harmonics.

    Returns
    -------
    y : array_like
        Evaluated series.
    """
    y = pars[0] * np.ones_like(phase)
    for i in range(order):
        y = y + pars[i * 2 + 1] * np.sin(2. * np.pi * (i + 1) * phase) \
              + pars[i * 2 + 2] * np.cos(2. * np.pi * (i + 1) * phase)

    return y


def _residuals(pars, phase, mag, order):
    return mag - fourier_series(pars, phase, order)


def fit_fourier(date, mag, period, order):
    """
    Least-squares fit of a Fourier series of a given order at a given period.

    Parameters
    ----------
    date : array_like
        Observed dates.
    mag : array_like
        Observed magnitudes.
    period : float
        Period to fold on.
    order : int
        Number of harmonics to fit.

    Returns
    -------
    params : array_like
        Fitted Fourier coefficients.
    phase : array_like
        Phase of each data point, in [0, 1).
    fitted : array_like
        Fitted magnitude at each data point.
    rss : float
        Residual sum of squares of the fit.
    """
    phase = (date % period) / period

    p0 = np.zeros(order * 2 + 1)
    p0[0] = np.mean(mag)

    params, success = leastsq(_residuals, p0, args=(phase, mag, order))
    fitted = fourier_series(params, phase, order)
    rss = float(np.sum((mag - fitted) ** 2))

    return params, phase, fitted, rss


def bic(n, k, rss):
    """
    Bayesian Information Criterion for a Gaussian-noise least-squares fit.

    Parameters
    ----------
    n : int
        Number of data points.
    k : int
        Number of fitted parameters.
    rss : float
        Residual sum of squares.

    Returns
    -------
    bic : float
        BIC value. Lower is better.
    """
    n = max(n, 1)
    rss = max(rss, 1e-12)

    return n * np.log(rss / n) + k * np.log(n)


def check_period_doubling(date, mag, period, order=4):
    """
    Decide whether the true period is `period` or `2 * period`.

    Parameters
    ----------
    date : array_like
        Observed dates.
    mag : array_like
        Observed magnitudes.
    period : float
        Candidate period found by Lomb-Scargle.
    order : int, optional
        Number of harmonics to use for the fit at `period`. The fit at
        `2 * period` uses `2 * order` harmonics (see module docstring).

    Returns
    -------
    result : dict
        All periods, fit statistics, the doubling decision, and the
        folded-phase/fitted arrays needed to plot both fits.
    """
    n = len(date)

    params_p, phase_p, fitted_p, rss_p = fit_fourier(date, mag, period, order)
    params_2p, phase_2p, fitted_2p, rss_2p = fit_fourier(
        date, mag, period * 2., order * 2)

    k_p = order * 2 + 1
    k_2p = order * 4 + 1

    bic_p = bic(n, k_p, rss_p)
    bic_2p = bic(n, k_2p, rss_2p)

    doubled = bool(bic_2p < bic_p)

    # Odd/even eclipse-depth diagnostic: mean magnitude in each phase half
    # of the 2P-folded curve. A large difference is visible evidence of two
    # unequal eclipses per true cycle.
    first_half = phase_2p < 0.5
    second_half = ~first_half
    odd_even_depth_diff = float(
        np.mean(mag[second_half]) - np.mean(mag[first_half]))

    return {
        'period': float(period),
        'period_doubled_candidate': float(period * 2.),
        'order': order,
        'n_points': n,
        'rss_period': rss_p,
        'rss_double_period': rss_2p,
        'k_period': k_p,
        'k_double_period': k_2p,
        'bic_period': float(bic_p),
        'bic_double_period': float(bic_2p),
        'doubled': doubled,
        'odd_even_depth_diff': odd_even_depth_diff,
        'phase_period': phase_p,
        'fitted_period': fitted_p,
        'phase_double_period': phase_2p,
        'fitted_double_period': fitted_2p,
    }


def save_diagnostics(date, mag, result, output_dir, star_id='star'):
    """
    Write a plot and a JSON report summarizing the P-vs-2P check.

    Requires matplotlib, imported lazily here so that normal feature
    extraction never needs it.

    Parameters
    ----------
    date : array_like
        Observed dates.
    mag : array_like
        Observed magnitudes.
    result : dict
        Output of `check_period_doubling`.
    output_dir : str
        Directory to write files into (created if missing).
    star_id : str, optional
        Used as the output filenames' prefix.

    Returns
    -------
    png_path : str
        Path to the written plot.
    json_path : str
        Path to the written JSON report.
    """
    import os
    import json

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(7, 9))

    panels = [
        (axes[0], 'phase_period', 'fitted_period', 'period',
         'rss_period', 'bic_period', 'P'),
        (axes[1], 'phase_double_period', 'fitted_double_period',
         'period_doubled_candidate', 'rss_double_period',
         'bic_double_period', '2P'),
    ]
    for ax, phase_key, fit_key, period_key, rss_key, bic_key, label in panels:
        phase = result[phase_key]
        order_sort = np.argsort(phase)

        ax.plot(phase[order_sort], mag[order_sort], '.',
                 ms=3, color='0.4', label='data')
        ax.plot(phase[order_sort], result[fit_key][order_sort], '-',
                 color='crimson', label='Fourier fit')
        ax.invert_yaxis()
        ax.set_xlabel('Phase')
        ax.set_ylabel('Magnitude')
        ax.set_title('%s = %.6f d   RSS=%.4g   BIC=%.2f'
                      % (label, result[period_key], result[rss_key],
                         result[bic_key]))
        ax.legend(loc='best', fontsize=8)

    decision = '2P' if result['doubled'] else 'P'
    fig.suptitle('%s -- period-doubling check -> chosen: %s'
                  % (star_id, decision))
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    png_path = os.path.join(output_dir, '%s_period_check.png' % star_id)
    fig.savefig(png_path, dpi=120)
    plt.close(fig)

    report = dict((k, v) for k, v in result.items()
                  if not isinstance(v, np.ndarray))
    report['decision'] = decision
    report['delta_bic'] = result['bic_period'] - result['bic_double_period']

    json_path = os.path.join(output_dir, '%s_period_check.json' % star_id)
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)

    return png_path, json_path
