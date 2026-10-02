"""
MD-DGFLS Evaluation Statistics Module (src/eval/statistics.py)
Implements statistical inference for research-grade reporting:
- Bootstrap Confidence Intervals (95% CI with B >= 1000 resamples)
- Paired Significance Testing (Wilcoxon signed-rank test)
- Multi-seed aggregation (mean ± std across seeds)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats


def compute_bootstrap_ci(
    data: Union[List[float], np.ndarray],
    num_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """
    Computes empirical bootstrap confidence interval using percentile resampling.

    Parameters:
    - data: 1D array of metric values (e.g. IoU scores across N images)
    - num_resamples: Number of bootstrap iterations (minimum 1000)
    - ci: Confidence level (default: 0.95)
    - seed: Random seed for deterministic reproducibility

    Returns:
    - dict with keys: mean, std, median, ci_lower, ci_upper, num_resamples, sample_size
    """
    arr = np.asarray(data, dtype=np.float64)
    # Filter out NaNs if any
    arr = arr[~np.isnan(arr)]
    n = len(arr)

    if n == 0:
        return {
            "mean": float(np.nan),
            "std": float(np.nan),
            "median": float(np.nan),
            "ci_lower": float(np.nan),
            "ci_upper": float(np.nan),
            "num_resamples": num_resamples,
            "sample_size": 0
        }

    if n == 1:
        val = float(arr[0])
        return {
            "mean": val,
            "std": 0.0,
            "median": val,
            "ci_lower": val,
            "ci_upper": val,
            "num_resamples": num_resamples,
            "sample_size": 1
        }

    rng = np.random.RandomState(seed)
    boot_indices = rng.randint(0, n, size=(num_resamples, n))
    boot_means = np.mean(arr[boot_indices], axis=1)

    alpha = (1.0 - ci) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    ci_lower = float(np.percentile(boot_means, lower_pct))
    ci_upper = float(np.percentile(boot_means, upper_pct))

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if n > 1 else 0.0,
        "median": float(np.median(arr)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "num_resamples": num_resamples,
        "sample_size": n
    }


def compute_paired_wilcoxon(
    scores_system: Union[List[float], np.ndarray],
    scores_baseline: Union[List[float], np.ndarray],
    alternative: str = "two-sided"
) -> Dict[str, Union[float, bool, str]]:
    """
    Performs Wilcoxon signed-rank test for paired sample comparison (e.g. MD-DGFLS vs Baseline).
    Tests the null hypothesis that the distribution of differences is symmetric about zero.

    Parameters:
    - scores_system: Per-image metric scores for primary system
    - scores_baseline: Per-image metric scores for baseline model on the exact same images
    - alternative: 'two-sided', 'greater', or 'less'

    Returns:
    - dict with keys: statistic, p_value, mean_diff, median_diff, significant_05, significant_01
    """
    s = np.asarray(scores_system, dtype=np.float64)
    b = np.asarray(scores_baseline, dtype=np.float64)

    if len(s) != len(b):
        raise ValueError(f"Sample length mismatch: system={len(s)}, baseline={len(b)}")

    # Filter invalid entries
    valid = (~np.isnan(s)) & (~np.isnan(b))
    s_clean = s[valid]
    b_clean = b[valid]
    n = len(s_clean)

    if n < 2:
        return {
            "statistic": float(np.nan),
            "p_value": float(np.nan),
            "mean_difference": float(np.nan),
            "median_difference": float(np.nan),
            "significant_at_05": False,
            "significant_at_01": False,
            "notes": "Insufficient samples (N < 2)"
        }

    diffs = s_clean - b_clean
    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))

    # If all differences are zero (identical models or zero errors)
    if np.all(diffs == 0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "mean_difference": 0.0,
            "median_difference": 0.0,
            "significant_at_05": False,
            "significant_at_01": False,
            "notes": "Identical paired performance across all samples"
        }

    try:
        res = stats.wilcoxon(diffs, alternative=alternative)
        stat = float(res.statistic)
        p_val = float(res.pvalue)
    except Exception as e:
        return {
            "statistic": float(np.nan),
            "p_value": float(np.nan),
            "mean_difference": mean_diff,
            "median_difference": median_diff,
            "significant_at_05": False,
            "significant_at_01": False,
            "notes": f"Wilcoxon computation failed: {str(e)}"
        }

    return {
        "statistic": stat,
        "p_value": p_val,
        "mean_difference": mean_diff,
        "median_difference": median_diff,
        "significant_at_05": bool(p_val < 0.05),
        "significant_at_01": bool(p_val < 0.01),
        "notes": "Success"
    }


def aggregate_over_seeds(
    seed_metrics: List[Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Computes mean and sample standard deviation across multiple seeds.

    Parameters:
    - seed_metrics: List of metric dicts (one per seed)

    Returns:
    - dict of {metric_name: {'mean': val, 'std': val, 'values': list}}
    """
    if not seed_metrics:
        return {}

    all_keys = list(seed_metrics[0].keys())
    aggregated = {}

    for k in all_keys:
        vals = [m[k] for m in seed_metrics if k in m and not np.isnan(m[k])]
        if vals:
            v_arr = np.asarray(vals, dtype=np.float64)
            aggregated[k] = {
                "mean": float(np.mean(v_arr)),
                "std": float(np.std(v_arr, ddof=1)) if len(v_arr) > 1 else 0.0,
                "values": [float(x) for x in v_arr]
            }
        else:
            aggregated[k] = {
                "mean": float(np.nan),
                "std": float(np.nan),
                "values": []
            }

    return aggregated
