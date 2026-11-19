"""
P vs 2P period-doubling check for eclipsing binaries, pulsator-safe.

A single-harmonic Lomb-Scargle fit (one sine + one cosine) locks onto half
the true period whenever an eclipsing binary's two eclipses are similar,
since one sinusoid per cycle fits the data better at twice the true
frequency. This module decides whether a period found that way should be
doubled.

The decision is *morphological*, not a bare goodness-of-fit comparison. A
multi-harmonic fit at 2P uses twice as many harmonics as the fit at P, so its
even harmonics reproduce the P model exactly and its odd harmonics are the
only terms that can express a difference between the two half-cycles -- i.e. a
primary-vs-secondary eclipse difference, or the O'Connell (unequal-maxima)
asymmetry of a contact binary. A genuine pulsator folded at 2P is just its
single pulse repeated twice, so its odd harmonics carry only noise.

A blanket ``BIC(2P) < BIC(P)`` rule fails in both directions on real data:
it spuriously doubles pulsators whose odd harmonics catch Blazhko / cycle-to-
cycle modulation (false positives), and it misses contact binaries whose
near-equal eclipses give an odd-harmonic signal too weak to overcome BIC's
parameter penalty (false negatives). Instead we score the decision from a few
interpretable statistics -- the noise-calibrated odd-harmonic SNR at 2P plus
light-curve shape features (Fourier ratios, skew, amplitude ratio) that tell a
pulsator shape apart from an eclipse shape -- with coefficients tuned on the
labelled LINEAR variable-star sample (see upsilon/test/evaluate_doubling.py).
"""

import numpy as np
import scipy.stats as ss
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
        Fitted Fourier coefficients [mean, a1, b1, a2, b2, ...].
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


def harmonic_amplitudes(params, order):
    """
    Per-harmonic amplitudes of a fitted Fourier series.

    Parameters
    ----------
    params : array_like
        Fourier coefficients [mean, a1, b1, a2, b2, ...].
    order : int
        Number of harmonics.

    Returns
    -------
    amp : numpy.ndarray
        Amplitude sqrt(a_k^2 + b_k^2) of harmonics k = 1 .. order.
    a : numpy.ndarray
        Sine coefficients a_k.
    b : numpy.ndarray
        Cosine coefficients b_k.
    """
    params = np.asarray(params, dtype=float)
    a = params[1:order * 2 + 1:2]
    b = params[2:order * 2 + 1:2]
    amp = np.hypot(a, b)
    return amp, a, b


def _half_mag_amplitude_ratio(mag, avg, weight):
    """Ratio of the faint-side to bright-side weighted scatter about `avg`.

    Larger for eclipse-dominated (dips below the baseline) light curves than
    for symmetric pulsators. Mirrors ExtractFeatures.half_mag_amplitude_ratio.
    """
    faint = mag > avg
    bright = ~faint
    if not np.any(faint) or not np.any(bright):
        return 1.0
    wf, wb = weight[faint], weight[bright]
    lower = np.sum((mag[faint] - avg) ** 2 * wf) / np.sum(wf)
    higher = np.sum((mag[bright] - avg) ** 2 * wb) / np.sum(wb)
    if higher <= 0:
        return 1.0
    return float(np.sqrt(lower / higher))


def decision_features(date, mag, period, order=4, err=None):
    """
    Fit Fourier models at P and 2P and derive the statistics used to decide
    whether `period` should be doubled.

    Parameters
    ----------
    date, mag : array_like
        Observed dates and magnitudes.
    period : float
        Candidate period found by Lomb-Scargle.
    order : int, optional
        Number of harmonics for the fit at `period`; the fit at `2 * period`
        uses `2 * order` harmonics.
    err : array_like, optional
        Magnitude errors, used to set the noise floor for the odd-harmonic
        SNR. If None, the P-fit residual RMS is used instead.

    Returns
    -------
    feats : dict
        Fit products (params/phase/fitted/rss/bic for P and 2P) plus the
        scalar decision features: delta_bic, r_odd, snr_odd, amp, r21, r31,
        phi21, skew, hl_amp_ratio.
    """
    date = np.asarray(date, dtype=float)
    mag = np.asarray(mag, dtype=float)
    n = len(date)

    params_p, phase_p, fitted_p, rss_p = fit_fourier(date, mag, period, order)
    params_2p, phase_2p, fitted_2p, rss_2p = fit_fourier(
        date, mag, period * 2., order * 2)

    k_p = order * 2 + 1
    k_2p = order * 4 + 1
    bic_p = bic(n, k_p, rss_p)
    bic_2p = bic(n, k_2p, rss_2p)

    # Harmonic decomposition at 2P. Even harmonics (2, 4, ...) reconstruct the
    # single-period (P) shape; odd harmonics (1, 3, ...) are the sub-harmonic
    # terms that express a difference between the two half-cycles.
    amp_2p, _, _ = harmonic_amplitudes(params_2p, order * 2)
    odd = amp_2p[0::2]   # harmonics 1, 3, 5, ...
    even = amp_2p[1::2]  # harmonics 2, 4, 6, ...
    a_odd = float(np.sqrt(np.sum(odd ** 2)))
    a_even = float(np.sqrt(np.sum(even ** 2)))
    r_odd = a_odd / (a_odd + a_even + 1e-12)

    # Noise floor for a single Fourier amplitude: sigma * sqrt(2 / n).
    if err is not None:
        sigma = float(np.median(np.asarray(err, dtype=float)))
    else:
        sigma = float(np.sqrt(rss_p / max(n, 1)))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(mag)) or 1e-6
    n_odd = len(odd)
    amp_noise = sigma * np.sqrt(2.0 / max(n, 1))
    snr_odd = a_odd / (amp_noise * np.sqrt(max(n_odd, 1)) + 1e-12)

    # Shape features from the P fit and the raw magnitudes.
    amp_p, a_p, b_p = harmonic_amplitudes(params_p, order)
    a1 = amp_p[0] if amp_p[0] > 0 else 1e-12
    r21 = float(amp_p[1] / a1) if order >= 2 else 0.0
    r31 = float(amp_p[2] / a1) if order >= 3 else 0.0
    phase1 = np.arctan2(b_p[0], a_p[0])
    phase2 = np.arctan2(b_p[1], a_p[1]) if order >= 2 else 0.0
    phi21 = float(np.arctan2(np.sin(phase2 - 2 * phase1),
                             np.cos(phase2 - 2 * phase1)))

    skew = float(ss.skew(mag))
    amplitude = float(np.percentile(mag, 95) - np.percentile(mag, 5))
    weight = 1.0 / np.asarray(err, dtype=float) if err is not None \
        else np.ones(n)
    weight = np.where(np.isfinite(weight) & (weight > 0), weight, 1.0)
    hl_amp_ratio = _half_mag_amplitude_ratio(mag, np.median(mag), weight)

    return {
        'n_points': n,
        'order': order,
        'params_p': params_p, 'phase_p': phase_p,
        'fitted_p': fitted_p, 'rss_p': rss_p, 'bic_p': float(bic_p),
        'params_2p': params_2p, 'phase_2p': phase_2p,
        'fitted_2p': fitted_2p, 'rss_2p': rss_2p, 'bic_2p': float(bic_2p),
        'k_p': k_p, 'k_2p': k_2p,
        'a_odd': a_odd, 'a_even': a_even,
        'delta_bic': float(bic_p - bic_2p),
        'r_odd': float(r_odd),
        'snr_odd': float(snr_odd),
        'amp': amplitude,
        'r21': r21, 'r31': r31, 'phi21': phi21,
        'skew': skew, 'hl_amp_ratio': float(hl_amp_ratio),
    }


# Feature order for the tuned linear decision score. The score is
#   s = intercept + sum_i coef[name_i] * feature_i
# and the period is doubled when s > 0. Coefficients already absorb feature
# standardization, so no scaler is needed at runtime. They are produced by
# upsilon/test/evaluate_doubling.py (`--tune`) on the labelled LINEAR sample.
_TUNED_FEATURES = ('delta_bic', 'snr_odd', 'r_odd', 'amp',
                   'r21', 'r31', 'phi21', 'skew', 'hl_amp_ratio')

# Tuned on the labelled LINEAR sample (150 stars/type, seed 0) with a balanced
# logistic regression, with the decision threshold folded into `intercept` so
# that the pulsator false-double rate is held near 2%. Reproduce / retune with
#   python -m upsilon.test.evaluate_doubling
# The score is dominated by delta_bic (evidence for a second feature per cycle)
# with morphology corrections: eclipse-like curves (positive skew, large
# hl_amp_ratio) double at a lower delta_bic, while low-amplitude and
# high-r_odd-but-weak pulsator shapes are suppressed. r21/r31/phi21 are left at
# zero: they conflate near-sinusoidal RRc pulsators with contact binaries and
# backfire if used here.
_TUNED_COEF = {
    'intercept': 0.981405,
    'delta_bic': 0.043243,
    'snr_odd': -4.46743e-06,
    'r_odd': -4.74359,
    'amp': -2.04076,
    'r21': 0.0,
    'r31': 0.0,
    'phi21': 0.0,
    'skew': 0.367128,
    'hl_amp_ratio': 0.121616,
}


def tuned_score(feats, coef=None):
    """Linear decision score; period is doubled when this is > 0."""
    coef = _TUNED_COEF if coef is None else coef
    s = coef.get('intercept', 0.0)
    for name in _TUNED_FEATURES:
        s += coef.get(name, 0.0) * feats[name]
    return float(s)


def check_period_doubling(date, mag, period, order=4, err=None, rule='tuned'):
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
        `2 * period` uses `2 * order` harmonics.
    err : array_like, optional
        Magnitude errors, used to set the odd-harmonic noise floor.
    rule : {'tuned', 'bic'}, optional
        'tuned' (default) uses the pulsator-safe morphology score; 'bic'
        reproduces the legacy ``BIC(2P) < BIC(P)`` rule (for comparison).

    Returns
    -------
    result : dict
        All periods, fit statistics, the decision, the morphology features,
        and the folded-phase/fitted arrays needed to plot both fits.
    """
    feats = decision_features(date, mag, period, order=order, err=err)

    score = tuned_score(feats)
    if rule == 'bic':
        doubled = bool(feats['bic_2p'] < feats['bic_p'])
    else:
        doubled = bool(score > 0.0)

    result = {
        'period': float(period),
        'period_doubled_candidate': float(period * 2.),
        'order': order,
        'n_points': feats['n_points'],
        'rss_period': feats['rss_p'],
        'rss_double_period': feats['rss_2p'],
        'k_period': feats['k_p'],
        'k_double_period': feats['k_2p'],
        'bic_period': feats['bic_p'],
        'bic_double_period': feats['bic_2p'],
        'delta_bic': feats['delta_bic'],
        'r_odd': feats['r_odd'],
        'snr_odd': feats['snr_odd'],
        'amp': feats['amp'],
        'r21': feats['r21'],
        'r31': feats['r31'],
        'phi21': feats['phi21'],
        'skew': feats['skew'],
        'hl_amp_ratio': feats['hl_amp_ratio'],
        'decision_score': score,
        'rule': rule,
        'doubled': doubled,
        'phase_period': feats['phase_p'],
        'fitted_period': feats['fitted_p'],
        'phase_double_period': feats['phase_2p'],
        'fitted_double_period': feats['fitted_2p'],
    }
    return result


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
    fig.suptitle('%s -- period-doubling check -> chosen: %s\n'
                 'snr_odd=%.1f  r_odd=%.2f  dBIC=%.1f  score=%.2f'
                 % (star_id, decision, result['snr_odd'], result['r_odd'],
                    result['delta_bic'], result['decision_score']))
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    png_path = os.path.join(output_dir, '%s_period_check.png' % star_id)
    fig.savefig(png_path, dpi=120)
    plt.close(fig)

    report = dict((k, v) for k, v in result.items()
                  if not isinstance(v, np.ndarray))
    report['decision'] = decision

    json_path = os.path.join(output_dir, '%s_period_check.json' % star_id)
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)

    return png_path, json_path
