"""Input definitions and the stock pool used by the reported strategy.

The external prepared panel contains these inputs in this exact order. Financial
inputs were built from daily report snapshots with report period, announcement
date and actual announcement date no later than the signal date. This module
does not reconstruct the vendor database or fill unavailable observations.
"""

import numpy as np


DEFINITIONS = {
    "F1": "Weighted recent-four-report ROE / its 60-session mean - 1; float32 absolute values <=1e-10 become zero",
    "F2": "120-session change in the four-report ROA slope / its 120-session population standard deviation",
    "F3": "120-session change in the four-report revenue-growth slope / its 120-session population standard deviation",
    "F4": "120-session change in the four-report operating-profit/equity slope / its 120-session population standard deviation (unused)",
    "F5": "120-session change in latest-report debt/equity / its 120-session population standard deviation",
    "T1": "Adjusted close[t] / adjusted close[t-20] - 1",
    "T2": "Adjusted close[t] / adjusted close[t-60] - 1",
    "T3": "20-session population standard deviation of close/preclose - 1",
    "T4": "Adjusted close / its 20-session mean - 1 (unused)",
    "T5": "20-session mean of finite, nonnegative turnover",
    "F6": "Latest visible parent equity in CNY / (market cap in ten-thousand CNY * 10000); retain negative equity",
    "F7": "Latest visible parent net-profit TTM / (market cap * 10000); retain losses; TTM=current YTD+prior FY-prior same quarter, each component separately visible; annual reports use current FY",
    "F8": "Latest visible annualized ROA, in percentage points",
    "F9": "Latest visible operating cash flow/revenue ratio",
    "F10": "Negative latest visible debt/assets, in percentage points",
    "T6": "Adjusted close[t-20] / adjusted close[t-126] - 1",
}

FEATURE_NAMES = list(DEFINITIONS)
CANDIDATE_NAMES = [
    "F1", "F2", "F3", "F5", "T5", "T3", "T1", "T2",
    "F6", "F7", "F8", "F9", "F10", "T6",
]


def feature_indices(names, feature_names=FEATURE_NAMES):
    """Map named inputs to the saved sixteen-column axis, preserving order."""
    return [feature_names.index(name) for name in names]


def common_eligible(raw_base_eligible, features, feature_names=FEATURE_NAMES,
                    candidate_names=CANDIDATE_NAMES):
    """Intersect original eligibility with all fourteen finite candidate inputs.

    F4 and T4 remain in the prepared sixteen-column array but do not restrict
    the opportunity set. This mask never depends on future-return availability.
    The explicit ndarray copy preserves the read-only parent memmap unchanged.
    """
    eligible = np.array(raw_base_eligible, dtype=bool, copy=True)
    for index in feature_indices(candidate_names, feature_names):
        eligible &= np.isfinite(features[:, :, index])
    return eligible
