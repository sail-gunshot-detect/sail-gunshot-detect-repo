"""
Wilcoxon Statistical Analysis for Gunshot Detection

This script performs robust statistical analysis of gunshot detection simulation results
using Wilcoxon signed-rank tests. It supports both per-run and per-file analyses,
with options for paired tests and one-sample tests against known baselines.

Usage:
    python wilcoxon.py --flat_csv results.csv --threshold 0.5 --outdir analysis_output
    python wilcoxon.py --perrun_csv metrics.csv --baseline_fpr 0.15 --alternative greater
"""

import argparse
import json
import os
import warnings
from scipy.stats import wilcoxon, ttest_rel, binom

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------- utility functions ----------------

def safe_wilcoxon_one_sample(data_in, null_value=0.0, alternative='greater'):
    """
    Perform one-sample Wilcoxon signed-rank test against a constant.

    Args:
        data_in: Array of data values
        null_value: Null hypothesis value to test against
        alternative: Alternative hypothesis ('greater', 'less', 'two-sided')

    Returns:
        tuple: (statistic, p_value, method_used, note)
    """
    data = np.asarray(data_in, dtype=float)
    data = data[~np.isnan(data)]
    n = len(data)

    if n == 0:
        return float('nan'), float('nan'), 'no_data', 'no data after removing NaNs'

    # Center data around null hypothesis
    centered = data - null_value
    nz = np.count_nonzero(centered)

    if nz == 0:
        return 0.0, 1.0, 'all_equal_null', 'all values equal to null value (p=1)'

    try:
        stat, p = wilcoxon(centered, alternative=alternative)
        return float(stat), float(p), 'wilcoxon_one_sample', 'one-sample wilcoxon succeeded'
    except (TypeError, ValueError):
        # Fallback to sign test
        pos = int(np.sum(centered > 0))
        neg = int(np.sum(centered < 0))
        n_nonzero = pos + neg

        if n_nonzero == 0:
            return 0.0, 1.0, 'all_zero_centered', 'no non-zero differences from null'

        if alternative == 'greater':
            p = float(binom.sf(pos - 1, n_nonzero, 0.5))
        elif alternative == 'less':
            p = float(binom.cdf(pos, n_nonzero, 0.5))
        else:
            k = min(pos, neg)
            p = float(2.0 * binom.cdf(k, n_nonzero, 0.5))
            p = min(p, 1.0)

        return float('nan'), float(p), 'sign_test_fallback', 'used sign test fallback'


def safe_wilcoxon_one_sided(a_in, b_in, alternative='greater'):
    """
    Robust paired Wilcoxon signed-rank test wrapper.

    Args:
        a_in: First array of paired data
        b_in: Second array of paired data
        alternative: Alternative hypothesis ('greater', 'less', 'two-sided')

    Returns:
        tuple: (statistic, p_value, method_used, note)
    """
    a = np.asarray(a_in, dtype=float)
    b = np.asarray(b_in, dtype=float)

    # Drop pairs with NaN
    mask = ~np.isnan(a) & ~np.isnan(b)
    a = a[mask]
    b = b[mask]
    n = len(a)

    if n == 0:
        return float('nan'), float('nan'), 'no_data', 'no paired data after removing NaNs'

    diff = a - b
    nz = np.count_nonzero(diff)

    if nz == 0:
        return 0.0, 1.0, 'all_zero', 'all paired differences are exactly zero (p=1)'

    # Try scipy wilcoxon with alternative parameter
    try:
        stat, p = wilcoxon(a, b, alternative=alternative)
        return float(stat), float(p), 'wilcoxon', 'standard wilcoxon succeeded'
    except TypeError:
        # Older scipy without alternative param
        stat, p_two = wilcoxon(a, b)
        mean_diff = float(np.mean(diff))

        if alternative == 'greater':
            p = p_two / 2.0 if mean_diff > 0 else 1.0 - p_two / 2.0
        elif alternative == 'less':
            p = p_two / 2.0 if mean_diff < 0 else 1.0 - p_two / 2.0
        else:
            p = p_two

        return float(stat), float(p), 'wilcoxon_converted', 'converted two-sided wilcoxon to one-sided'
    except ValueError:
        # Fallback to binomial sign test
        pos = int(np.sum(diff > 0))
        neg = int(np.sum(diff < 0))
        n_nonzero = pos + neg

        if n_nonzero == 0:
            return 0.0, 1.0, 'all_zero_after_mask', 'no non-zero diffs after mask (p=1)'

        if alternative == 'greater':
            p = float(binom.sf(pos - 1, n_nonzero, 0.5))
        elif alternative == 'less':
            p = float(binom.cdf(pos, n_nonzero, 0.5))
        else:
            k = min(pos, neg)
            p = float(2.0 * binom.cdf(k, n_nonzero, 0.5))
            p = min(p, 1.0)

        return float('nan'), float(p), 'binom_fallback', 'Wilcoxon failed; used binomial sign test fallback'


def bootstrap_mean_ci(arr, n_boot=5000, ci=95, seed=123):
    """
    Bootstrap confidence interval for mean.

    Args:
        arr: Input array
        n_boot: Number of bootstrap samples
        ci: Confidence interval percentage
        seed: Random seed

    Returns:
        tuple: (mean, lower_ci, upper_ci)
    """
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return float('nan'), float('nan'), float('nan')

    rng = np.random.RandomState(seed)
    boots = []
    n = len(arr)

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boots.append(np.mean(arr[idx]))

    lo, hi = np.percentile(boots, [(100-ci)/2.0, 100 - (100-ci)/2.0])
    return float(np.mean(arr)), float(lo), float(hi)


def paired_cohens_d(a, b):
    """Calculate paired Cohen's d effect size."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    d = a[mask] - b[mask]

    if len(d) < 2:
        return float('nan')

    sd = np.std(d, ddof=1)
    if sd == 0:
        return float('nan')

    return float(np.mean(d) / sd)


def plot_hist(arr, title, outpath, bins=25):
    """Create and save histogram plot."""
    plt.figure(figsize=(6,4))
    plt.hist(arr, bins=bins, edgecolor='k', alpha=0.85)
    plt.axvline(0, color='k', linestyle='--')
    plt.title(title)
    plt.xlabel('Difference')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_boxplot(arrs, labels, outpath):
    """Create and save boxplot."""
    plt.figure(figsize=(6,4))
    plt.boxplot(arrs, labels=labels)
    plt.ylabel('FPR')
    plt.title('Per-run FPR distributions')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -------------- core processing functions ----------------

def is_flattened_df(df):
    """Check if DataFrame contains flattened CSV columns."""
    flattened_signs = {'filepath', 'p1', 'p2', 'p3', 'run'}
    return len(flattened_signs.intersection(set(df.columns))) >= 3


def compute_per_run_metrics_from_flat(df_flat, threshold):
    """
    Compute per-run FPRs from flattened CSV DataFrame.

    Args:
        df_flat: DataFrame with flattened simulation results
        threshold: Classification threshold

    Returns:
        DataFrame: Per-run metrics
    """
    required = {'run','filepath','label','p0','p1','p2','p3','sail_pred','fancy_pred'}
    if not required.issubset(set(df_flat.columns)):
        missing = required - set(df_flat.columns)
        raise ValueError(f"Flattened CSV missing columns: {missing}")

    df = df_flat.copy()

    # Cast to numeric
    for c in ['p0','p1','p2','p3','sail_pred','fancy_pred','label','run']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Compute scores and flags
    df['single_score'] = df[['p1','p2','p3']].mean(axis=1)
    df['inf_flag'] = (df['p0'] >= threshold).astype(int)
    df['single_flag'] = (df['single_score'] >= threshold).astype(int)
    df['sail_flag'] = (df['sail_pred'] >= threshold).astype(int)
    df['fancy_flag'] = (df['fancy_pred'] >= threshold).astype(int)

    # Aggregate by run for negatives only (FPR computation)
    neg = df[df['label'] == 0]
    if len(neg) == 0:
        raise ValueError("No negative files found in flattened CSV; cannot compute per-run FPR.")

    group = neg.groupby('run').agg({
        'inf_flag':'mean',
        'single_flag':'mean',
        'sail_flag':'mean',
        'fancy_flag':'mean'
    }).rename(columns={
        'inf_flag':'inf_fpr',
        'single_flag':'single_fpr',
        'sail_flag':'sail_fpr',
        'fancy_flag':'fancy_fpr'
    }).reset_index()

    return group.sort_values('run').reset_index(drop=True)


# -------------- analysis functions ----------------

def run_per_run_tests(per_run_df, outdir, n_boot=5000, alternative='greater', baseline_fpr=None):
    """
    Run per-run statistical tests.

    Args:
        per_run_df: DataFrame with per-run metrics
        outdir: Output directory
        n_boot: Number of bootstrap samples
        alternative: Alternative hypothesis
        baseline_fpr: Known baseline FPR for one-sample tests

    Returns:
        dict: Test results
    """
    os.makedirs(outdir, exist_ok=True)

    # Ensure numeric types
    for c in ['inf_fpr','sail_fpr','fancy_fpr']:
        if c not in per_run_df.columns:
            raise ValueError(f"Missing column '{c}' in per_run_df")
        per_run_df[c] = pd.to_numeric(per_run_df[c], errors='coerce')

    inf = per_run_df['inf_fpr'].values
    sail = per_run_df['sail_fpr'].values
    fancy = per_run_df['fancy_fpr'].values

    # Create validity masks
    valid_mask_sail = (~np.isnan(inf)) & (~np.isnan(sail))
    valid_mask_fancy = (~np.isnan(inf)) & (~np.isnan(fancy))

    out = {}

    # Test: inf vs sail
    if np.sum(valid_mask_sail) < 2:
        out['inf_vs_sail'] = {'note':'insufficient_runs', 'n_valid': int(np.sum(valid_mask_sail))}
    else:
        stat_s, p_s, alt_s, note_s = safe_wilcoxon_one_sided(
            inf[valid_mask_sail], sail[valid_mask_sail], alternative=alternative
        )
        delta_s = (inf - sail)[valid_mask_sail]
        mean_s, lo_s, hi_s = bootstrap_mean_ci(delta_s, n_boot=n_boot, ci=95, seed=101)
        d_s = paired_cohens_d(inf[valid_mask_sail], sail[valid_mask_sail])

        try:
            tstat_s, tp_s = ttest_rel(inf[valid_mask_sail], sail[valid_mask_sail])
        except Exception:
            tstat_s, tp_s = float('nan'), float('nan')

        out['inf_vs_sail'] = {
            'n_valid_runs': int(np.sum(valid_mask_sail)),
            'mean_delta': float(mean_s),
            'bootstrap_95ci': [float(lo_s), float(hi_s)],
            'wilcoxon_stat': stat_s,
            'wilcoxon_p_one_sided': float(p_s),
            'wilcoxon_method': alt_s,
            'wilcoxon_note': note_s,
            'cohens_d_paired': float(d_s),
            'ttest_stat': float(tstat_s),
            'ttest_p_two_sided': float(tp_s)
        }
        plot_hist(delta_s, 'inf_fpr - sail_fpr (per-run)',
                 os.path.join(outdir, 'perrun_delta_inf_minus_sail_hist.png'))

    # Test: inf vs fancy
    if np.sum(valid_mask_fancy) < 2:
        out['inf_vs_fancy'] = {'note':'insufficient_runs', 'n_valid': int(np.sum(valid_mask_fancy))}
    else:
        stat_f, p_f, alt_f, note_f = safe_wilcoxon_one_sided(
            inf[valid_mask_fancy], fancy[valid_mask_fancy], alternative=alternative
        )
        delta_f = (inf - fancy)[valid_mask_fancy]
        mean_f, lo_f, hi_f = bootstrap_mean_ci(delta_f, n_boot=n_boot, ci=95, seed=102)
        d_f = paired_cohens_d(inf[valid_mask_fancy], fancy[valid_mask_fancy])

        try:
            tstat_f, tp_f = ttest_rel(inf[valid_mask_fancy], fancy[valid_mask_fancy])
        except Exception:
            tstat_f, tp_f = float('nan'), float('nan')

        out['inf_vs_fancy'] = {
            'n_valid_runs': int(np.sum(valid_mask_fancy)),
            'mean_delta': float(mean_f),
            'bootstrap_95ci': [float(lo_f), float(hi_f)],
            'wilcoxon_stat': stat_f,
            'wilcoxon_p_one_sided': float(p_f),
            'wilcoxon_method': alt_f,
            'wilcoxon_note': note_f,
            'cohens_d_paired': float(d_f),
            'ttest_stat': float(tstat_f),
            'ttest_p_two_sided': float(tp_f)
        }
        plot_hist(delta_f, 'inf_fpr - fancy_fpr (per-run)',
                 os.path.join(outdir, 'perrun_delta_inf_minus_fancy_hist.png'))

    # Create boxplot if possible
    try:
        inf_clean = per_run_df['inf_fpr'].dropna().values
        sail_clean = per_run_df['sail_fpr'].dropna().values
        fancy_clean = per_run_df['fancy_fpr'].dropna().values
        plot_boxplot([inf_clean, sail_clean, fancy_clean], ['inf','sail','fancy'],
                    os.path.join(outdir, 'perrun_fpr_boxplot.png'))
    except Exception:
        pass

    # One-sample tests against baseline
    if baseline_fpr is not None:
        out['one_sample_tests'] = {}

        # Test sail vs baseline
        sail_clean = per_run_df['sail_fpr'].dropna().values
        if len(sail_clean) > 0:
            stat_s, p_s, alt_s, note_s = safe_wilcoxon_one_sample(
                sail_clean, baseline_fpr, alternative='less'
            )
            mean_s, lo_s, hi_s = bootstrap_mean_ci(sail_clean, n_boot=n_boot, ci=95, seed=301)

            out['one_sample_tests']['sail_vs_baseline'] = {
                'baseline_fpr': float(baseline_fpr),
                'sail_mean_fpr': float(mean_s),
                'sail_bootstrap_95ci': [float(lo_s), float(hi_s)],
                'wilcoxon_stat': stat_s,
                'wilcoxon_p_one_sided_less': float(p_s),
                'wilcoxon_method': alt_s,
                'wilcoxon_note': note_s,
                'interpretation': 'p < 0.05 means sail significantly REDUCES FPR vs baseline'
            }

        # Test fancy vs baseline
        fancy_clean = per_run_df['fancy_fpr'].dropna().values
        if len(fancy_clean) > 0:
            stat_f, p_f, alt_f, note_f = safe_wilcoxon_one_sample(
                fancy_clean, baseline_fpr, alternative='less'
            )
            mean_f, lo_f, hi_f = bootstrap_mean_ci(fancy_clean, n_boot=n_boot, ci=95, seed=302)

            out['one_sample_tests']['fancy_vs_baseline'] = {
                'baseline_fpr': float(baseline_fpr),
                'fancy_mean_fpr': float(mean_f),
                'fancy_bootstrap_95ci': [float(lo_f), float(hi_f)],
                'wilcoxon_stat': stat_f,
                'wilcoxon_p_one_sided_less': float(p_f),
                'wilcoxon_method': alt_f,
                'wilcoxon_note': note_f,
                'interpretation': 'p < 0.05 means fancy significantly REDUCES FPR vs baseline'
            }

    # Save per-run metrics
    per_run_df.to_csv(os.path.join(outdir, 'per_run_metrics_used.csv'), index=False)
    return out


def run_per_file_tests_from_flat(flat_df, threshold, outdir, n_boot=5000, alternative='greater'):
    """
    Run per-file statistical tests.

    Args:
        flat_df: Flattened DataFrame with per-file results
        threshold: Classification threshold
        outdir: Output directory
        n_boot: Number of bootstrap samples
        alternative: Alternative hypothesis

    Returns:
        tuple: (results_dict, per_file_dataframe)
    """
    os.makedirs(outdir, exist_ok=True)

    # Validate and prepare data
    required = {'run','filepath','label','p0','p1','p2','p3','sail_pred','fancy_pred'}
    if not required.issubset(set(flat_df.columns)):
        missing = required - set(flat_df.columns)
        raise ValueError(f"Flattened CSV missing columns: {missing}")

    df = flat_df.copy()
    for c in ['p0','p1','p2','p3','sail_pred','fancy_pred','label','run']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Compute scores and flags
    df['single_score'] = df[['p1','p2','p3']].mean(axis=1)
    df['single_flag'] = (df['single_score'] >= threshold).astype(int)
    df['sail_flag'] = (df['sail_pred'] >= threshold).astype(int)
    df['fancy_flag'] = (df['fancy_pred'] >= threshold).astype(int)

    # Group by filepath and aggregate
    grouped = df.groupby('filepath')
    rows = []

    for fp, g in grouped:
        label_vals = g['label'].dropna().unique()
        if len(label_vals) == 0:
            continue
        label = int(label_vals[0])
        n_runs = int(len(g))
        fp_single = float(g['single_flag'].mean())
        fp_sail = float(g['sail_flag'].mean())
        fp_fancy = float(g['fancy_flag'].mean())
        mean_single_score = float(g['single_score'].mean())
        mean_sail_score = float(g['sail_pred'].mean())
        mean_fancy_score = float(g['fancy_pred'].mean())

        rows.append([fp, label, n_runs, fp_single, fp_sail, fp_fancy,
                    mean_single_score, mean_sail_score, mean_fancy_score])

    per_file_df = pd.DataFrame(rows, columns=[
        'filepath','label','n_runs','fp_single','fp_sail','fp_fancy',
        'mean_single_score','mean_sail_score','mean_fancy_score'
    ])

    # Test on negatives only
    negs = per_file_df[per_file_df['label'] == 0]
    if len(negs) == 0:
        raise ValueError("No negative files found (label==0). Cannot run per-file Wilcoxon.")

    out = {}

    # Test: single vs sail
    stat_s, p_s, alt_s, note_s = safe_wilcoxon_one_sided(
        negs['fp_single'].values, negs['fp_sail'].values, alternative=alternative
    )
    d_s = (negs['fp_single'].values - negs['fp_sail'].values)
    mean_s, lo_s, hi_s = bootstrap_mean_ci(d_s, n_boot=n_boot, ci=95, seed=201)
    cohens_s = paired_cohens_d(negs['fp_single'].values, negs['fp_sail'].values)

    out['single_vs_sail'] = {
        'n_neg_files': int(len(negs)),
        'mean_diff': float(mean_s),
        'bootstrap_95ci': [float(lo_s), float(hi_s)],
        'wilcoxon_stat': stat_s,
        'wilcoxon_p_one_sided': float(p_s),
        'wilcoxon_method': alt_s,
        'wilcoxon_note': note_s,
        'cohens_d_paired': float(cohens_s)
    }
    plot_hist(d_s, 'Per-file: fp_single - fp_sail (negatives)',
             os.path.join(outdir, 'perfile_delta_single_minus_sail_hist.png'))

    # Test: single vs fancy
    stat_f, p_f, alt_f, note_f = safe_wilcoxon_one_sided(
        negs['fp_single'].values, negs['fp_fancy'].values, alternative=alternative
    )
    d_f = (negs['fp_single'].values - negs['fp_fancy'].values)
    mean_f, lo_f, hi_f = bootstrap_mean_ci(d_f, n_boot=n_boot, ci=95, seed=202)
    cohens_f = paired_cohens_d(negs['fp_single'].values, negs['fp_fancy'].values)

    out['single_vs_fancy'] = {
        'n_neg_files': int(len(negs)),
        'mean_diff': float(mean_f),
        'bootstrap_95ci': [float(lo_f), float(hi_f)],
        'wilcoxon_stat': stat_f,
        'wilcoxon_p_one_sided': float(p_f),
        'wilcoxon_method': alt_f,
        'wilcoxon_note': note_f,
        'cohens_d_paired': float(cohens_f)
    }
    plot_hist(d_f, 'Per-file: fp_single - fp_fancy (negatives)',
             os.path.join(outdir, 'perfile_delta_single_minus_fancy_hist.png'))

    # Save per-file table
    per_file_df.to_csv(os.path.join(outdir, 'per_file_table.csv'), index=False)
    return out, per_file_df


# ----------------- main CLI ---------------------

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Statistical Analysis for Gunshot Detection Simulations'
    )
    parser.add_argument(
        "--perrun_csv", type=str, default=None,
        help="Per-run metrics CSV (inf_fpr, sail_fpr, fancy_fpr)"
    )
    parser.add_argument(
        "--flat_csv", type=str, default=None,
        help="Flattened per-run-per-file CSV"
    )
    parser.add_argument(
        "--outdir", type=str, default="analysis_results",
        help="Output directory"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold"
    )
    parser.add_argument(
        "--n_boot", type=int, default=5000,
        help="Number of bootstrap samples"
    )
    parser.add_argument(
        "--alternative", type=str, default="greater",
        choices=['greater','less','two-sided'],
        help="Wilcoxon alternative hypothesis"
    )
    parser.add_argument(
        "--baseline_fpr", type=float, default=None,
        help="Known baseline FPR for one-sample tests"
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    overall_summary = {
        'outdir': args.outdir,
        'threshold': args.threshold
    }

    # Load input data
    flat_df = None
    per_run_df = None

    if args.flat_csv:
        if not os.path.exists(args.flat_csv):
            raise FileNotFoundError(f"Flattened CSV not found: {args.flat_csv}")
        flat_df = pd.read_csv(args.flat_csv)
        overall_summary['flat_csv'] = args.flat_csv

    if args.perrun_csv:
        if not os.path.exists(args.perrun_csv):
            raise FileNotFoundError(f"Per-run CSV not found: {args.perrun_csv}")
        per_run_df = pd.read_csv(args.perrun_csv)
        overall_summary['perrun_csv'] = args.perrun_csv

    # If only flattened provided, compute per-run metrics
    if (per_run_df is None) and (flat_df is not None):
        print("Computing per-run metrics from flattened CSV...")
        per_run_df = compute_per_run_metrics_from_flat(flat_df, args.threshold)
        per_run_df.to_csv(
            os.path.join(args.outdir, 'per_run_metrics_computed_from_flattened.csv'),
            index=False
        )
        overall_summary['per_run_metrics_computed'] = True

    # Validate inputs
    if (per_run_df is None) and (flat_df is None):
        raise ValueError("Must supply at least one of --perrun_csv or --flat_csv")

    # Run analyses
    per_run_summary = None
    if per_run_df is not None:
        print("Running per-run paired tests...")
        per_run_summary = run_per_run_tests(
            per_run_df, args.outdir, n_boot=args.n_boot,
            alternative=args.alternative, baseline_fpr=args.baseline_fpr
        )
        overall_summary['per_run_summary'] = per_run_summary

    per_file_summary = None
    per_file_df = None
    if flat_df is not None:
        print("Running per-file paired Wilcoxon tests...")
        per_file_summary, per_file_df = run_per_file_tests_from_flat(
            flat_df, args.threshold, args.outdir,
            n_boot=args.n_boot, alternative=args.alternative
        )
        overall_summary['per_file_summary'] = per_file_summary

    # Save overall summary
    with open(os.path.join(args.outdir, 'overall_analysis_summary.json'), 'w') as f:
        json.dump(overall_summary, f, indent=2)

    print("Saved overall_analysis_summary.json")
    print("Done.")


if __name__ == "__main__":
    main()
