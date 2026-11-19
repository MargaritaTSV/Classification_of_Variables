"""
Evaluate and tune the P-vs-2P period-doubling decision on real LINEAR data.

The repo ships ~7010 LINEAR light curves (``allLINEARfinal_dat.tar.gz``) and a
catalog with visual classifications and periods
(``LINEARattributesFinalApr2013.dat``). LCtype legend: 1=RRab, 2=RRc, 4=EA
(Algol EB), 5=EW (contact EB), 6=delta Sct. Types 1, 2, 6 are pulsators whose
period must never be doubled; 4, 5 are eclipsing binaries, the case the
doubling check must catch.

Two decision scenarios are built per star from the catalog period P_true:
  * "atP"    : candidate = P_true        -> correct answer: do NOT double
  * "atHalf" : candidate = P_true / 2    -> correct answer: DO double
               (only for EBs; simulates Lomb-Scargle locking onto half P)

Usage (run from the repo root):
  python -m upsilon.test.evaluate_doubling              # report + tune
  python -m upsilon.test.evaluate_doubling --per-type 150 --seed 0
  python -m upsilon.test.evaluate_doubling --endtoend 40   # real LS pipeline
"""

import argparse
import io
import json
import os
import tarfile
import warnings

import numpy as np

from upsilon.extract_features import period_doubling as pdbl

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TARBALL = os.path.join(HERE, 'allLINEARfinal_dat.tar.gz')
CATALOG = os.path.join(HERE, 'LINEARattributesFinalApr2013.dat')
DIAG_DIR = os.path.join(HERE, 'diagnostics')

TYPE_LABEL = {1: 'RRab', 2: 'RRc', 4: 'EA', 5: 'EW', 6: 'dSct'}
PULSATORS = (1, 2, 6)
BINARIES = (4, 5)


def load_catalog():
    """objID (str) -> (lctype int, P_true float)."""
    cat = {}
    with open(CATALOG) as f:
        for line in f:
            if line.startswith('#'):
                continue
            p = line.split()
            if len(p) < 14:
                continue
            cat[p[13]] = (int(p[12]), 10.0 ** float(p[6]))
    return cat


def load_lightcurve(raw):
    """Parse MJD, mag, err from a LINEAR .dat file; clean bad points."""
    data = np.loadtxt(io.BytesIO(raw))
    if data.ndim != 2 or data.shape[1] < 3:
        return None
    date, mag, err = data[:, 0], data[:, 1], data[:, 2]
    good = (mag < 99.999) & np.isfinite(mag) & np.isfinite(date) & (err > 0)
    return date[good], mag[good], err[good]


def iter_sample(per_type, seed, types=(1, 2, 4, 5, 6), min_points=40):
    """Yield (oid, lctype, P_true, date, mag, err), stratified by type."""
    cat = load_catalog()
    rng = np.random.RandomState(seed)
    tar = tarfile.open(TARBALL)
    members = sorted((m for m in tar.getmembers() if m.isfile()),
                     key=lambda m: m.name)
    # Deterministic shuffle so a given seed always picks the same stars.
    order = rng.permutation(len(members))
    want = {t: per_type for t in types}
    got = {t: 0 for t in types}
    for idx in order:
        if all(got[t] >= want[t] for t in types):
            break
        m = members[idx]
        oid = m.name.replace('.dat', '')
        if oid not in cat:
            continue
        lt, p_true = cat[oid]
        if lt not in want or got[lt] >= want[lt]:
            continue
        lc = load_lightcurve(tar.extractfile(m).read())
        if lc is None or len(lc[0]) < min_points:
            continue
        got[lt] += 1
        yield oid, lt, p_true, lc[0], lc[1], lc[2]


def build_scenarios(per_type, seed):
    """Collect decision features + labels for every scenario in the sample.

    Returns a list of dicts: {oid, lctype, scenario, should_double, feats}.
    """
    rows = []
    for oid, lt, p_true, date, mag, err in iter_sample(per_type, seed):
        # atP: candidate is the true period -> must not double.
        feats = pdbl.decision_features(date, mag, p_true, err=err)
        rows.append(dict(oid=oid, lctype=lt, scenario='atP',
                         should_double=False, feats=feats))
        # atHalf: EB fed half the true period -> must double.
        if lt in BINARIES:
            feats = pdbl.decision_features(date, mag, p_true / 2.0, err=err)
            rows.append(dict(oid=oid, lctype=lt, scenario='atHalf',
                             should_double=True, feats=feats))
    return rows


def _decide(feats, coef, rule):
    if rule == 'bic':
        return feats['bic_2p'] < feats['bic_p']
    return pdbl.tuned_score(feats, coef) > 0.0


def per_type_report(rows, coef, rule, title):
    """Print false-double and EB-recovery rates per type for a decision rule."""
    print('\n== %s ==' % title)
    print('%-6s | n(atP) false-double | n(atHalf) EB-recovery' % 'type')
    print('-' * 56)
    overall_fp = overall_fp_n = overall_rec = overall_rec_n = 0
    for lt in (1, 2, 6, 4, 5):
        atP = [r for r in rows if r['lctype'] == lt and r['scenario'] == 'atP']
        atH = [r for r in rows
               if r['lctype'] == lt and r['scenario'] == 'atHalf']
        fp = sum(_decide(r['feats'], coef, rule) for r in atP)
        rec = sum(_decide(r['feats'], coef, rule) for r in atH)
        fp_s = '%3d/%-3d (%5.1f%%)' % (
            fp, len(atP), 100 * fp / max(len(atP), 1))
        rec_s = '%3d/%-3d (%5.1f%%)' % (
            rec, len(atH), 100 * rec / max(len(atH), 1)) if atH else '   --'
        print('%-6s | %-19s | %s' % (TYPE_LABEL[lt], fp_s, rec_s))
        overall_fp += fp
        overall_fp_n += len(atP)
        overall_rec += rec
        overall_rec_n += len(atH)
    print('-' * 56)
    print('%-6s | %3d/%-3d (%5.1f%%)      | %3d/%-3d (%5.1f%%)' % (
        'ALL', overall_fp, overall_fp_n, 100 * overall_fp / max(overall_fp_n, 1),
        overall_rec, overall_rec_n, 100 * overall_rec / max(overall_rec_n, 1)))
    return dict(false_double=overall_fp, false_double_n=overall_fp_n,
                recovery=overall_rec, recovery_n=overall_rec_n)


def tune(rows, seed):
    """Fit a logistic-regression decision rule; return runtime coefficients.

    Standardization is folded into the returned coefficients so no scaler is
    needed at runtime (score = intercept + sum coef_i * feature_i).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    names = list(pdbl._TUNED_FEATURES)
    X = np.array([[r['feats'][n] for n in names] for r in rows], dtype=float)
    y = np.array([1 if r['should_double'] else 0 for r in rows])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)

    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd[sd == 0] = 1.0
    clf = LogisticRegression(class_weight='balanced', max_iter=5000, C=1.0)
    clf.fit((X_tr - mu) / sd, y_tr)

    w_std = clf.coef_[0]
    b_std = float(clf.intercept_[0])
    # Unstandardize: s = b_std + sum w_std * (x - mu) / sd
    w = w_std / sd
    b = b_std - float(np.sum(w_std * mu / sd))

    coef = {'intercept': b}
    for n, wi in zip(names, w):
        coef[n] = float(wi)

    train_acc = clf.score((X_tr - mu) / sd, y_tr)
    test_acc = clf.score((X_te - mu) / sd, y_te)
    print('\n== Tuned logistic-regression decision rule ==')
    print('train acc %.3f   held-out test acc %.3f   (n=%d, pos=%d)'
          % (train_acc, test_acc, len(y), int(y.sum())))
    print('feature weights (standardized |w|, larger = more influential):')
    for n, wi in sorted(zip(names, w_std), key=lambda t: -abs(t[1])):
        print('   %-14s % .3f' % (n, wi))
    print('\nRuntime coefficients (paste into period_doubling._TUNED_COEF):')
    print('_TUNED_COEF = {')
    print("    'intercept': %.6g," % coef['intercept'])
    for n in names:
        print("    '%s': %.6g," % (n, coef[n]))
    print('}')
    return coef


def run_endtoend(n_per_type, seed, rule='tuned'):
    """Run the full ExtractFeatures pipeline (real Lomb-Scargle) and compare
    the final period to the catalog period, per type."""
    import time
    from upsilon.extract_features.extract_features import ExtractFeatures

    # Force the doubling rule used inside the pipeline (for before/after runs).
    _orig = pdbl.check_period_doubling
    pdbl.check_period_doubling = (
        lambda *a, **k: _orig(*a, **dict(k, rule=rule)))

    print('\n== End-to-end pipeline (real Lomb-Scargle, rule=%s) ==' % rule)
    t0 = time.time()
    stats = {t: dict(n=0, acc=0, raw_acc=0, doubled=0, false_dbl=0)
             for t in (1, 2, 4, 5, 6)}
    for oid, lt, p_true, date, mag, err in iter_sample(n_per_type, seed):
        try:
            ef = ExtractFeatures(date, mag, err, n_threads=1)
            ef.shallow_run()
            ef.get_period_LS(ef.date, ef.mag, ef.n_threads, ef.min_period)
        except Exception as exc:
            print('  skip %s (%s)' % (oid, exc))
            continue
        s = stats[lt]
        s['n'] += 1

        def matches(p):
            return abs(p - p_true) / p_true < 0.01 or \
                   abs(2 * p - p_true) / p_true < 0.01 or \
                   abs(p / 2 - p_true) / p_true < 0.01
        if abs(ef.period - p_true) / p_true < 0.01:
            s['acc'] += 1
        if abs(ef.ls_raw_period - p_true) / p_true < 0.01:
            s['raw_acc'] += 1
        if ef.period_doubled:
            s['doubled'] += 1
            # A double was wrong if the raw LS period already matched P_true.
            if abs(ef.ls_raw_period - p_true) / p_true < 0.01:
                s['false_dbl'] += 1

    print('%-6s | n   | final P == P_true | doubled | bad-double' % 'type')
    print('-' * 60)
    for lt in (1, 2, 6, 4, 5):
        s = stats[lt]
        if s['n'] == 0:
            continue
        print('%-6s | %-3d | %3d (%5.1f%%)      | %3d     | %d'
              % (TYPE_LABEL[lt], s['n'], s['acc'],
                 100 * s['acc'] / s['n'], s['doubled'], s['false_dbl']))
    print('(ran in %.1f s)' % (time.time() - t0))
    pdbl.check_period_doubling = _orig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--per-type', type=int, default=150)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--endtoend', type=int, default=0,
                    help='run the real LS pipeline on N stars per type')
    ap.add_argument('--e2e-rule', choices=('tuned', 'bic'), default='tuned',
                    help='doubling rule to use in the end-to-end pipeline')
    ap.add_argument('--no-tune', action='store_true')
    args = ap.parse_args()

    if args.endtoend:
        run_endtoend(args.endtoend, args.seed, rule=args.e2e_rule)
        return

    print('Building decision scenarios (%d per type, seed %d)...'
          % (args.per_type, args.seed))
    rows = build_scenarios(args.per_type, args.seed)
    print('%d scenarios (%d should-double).'
          % (len(rows), sum(r['should_double'] for r in rows)))

    base = per_type_report(rows, pdbl._TUNED_COEF, 'bic',
                           'BASELINE: legacy BIC(2P) < BIC(P) rule')
    if not args.no_tune:
        coef = tune(rows, args.seed)
        tuned = per_type_report(rows, coef, 'tuned',
                                'TUNED: morphology decision rule')
        os.makedirs(DIAG_DIR, exist_ok=True)
        with open(os.path.join(DIAG_DIR, 'doubling_eval.json'), 'w') as f:
            json.dump(dict(per_type=args.per_type, seed=args.seed,
                           baseline=base, tuned=tuned, coef=coef), f, indent=2)
        print('\nWrote diagnostics/doubling_eval.json')


if __name__ == '__main__':
    main()
