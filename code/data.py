"""Read the external prepared panel; no data download or vendor reconstruction.

Only .npy market/label arrays, codes.json, the sixteen-input .npy panel and an
external corporate-action CSV are required. Research caches, saved predictions
and the former project's Python modules are not used.
"""

import csv
import json
from pathlib import Path

import numpy as np

from factors import CANDIDATE_NAMES, FEATURE_NAMES, common_eligible


PANEL_ARRAYS = (
    "open", "close", "preclose", "adjusted_open", "adjusted_close",
    "upper", "lower", "volume", "status", "universe", "industry", "mcap",
    "forward_returns", "base_eligible",
)
ACTION_COLUMNS = (
    "old_code", "new_code", "announcement_date", "effective_date",
    "listing_date", "ratio", "cash_price", "payment_date", "source_url",
    "settlement_source_url", "modelling_assumption",
)


def load_data(data_dir, expanded_features_path, actions_path, config):
    """Return read-only prepared arrays and the identical fourteen-input pool.

    ``data_dir`` contains seventeen .npy files and codes.json. The separate
    expanded array has shape (date, stock, 16), dtype float32. Prepared labels
    are next-open to open twenty sessions later: open[t+21]/open[t+1]-1,
    using the causal adjusted-price chain. Missing features/labels stay NaN.
    """
    folder = Path(data_dir)
    if config["horizon"] != 20:
        raise ValueError("The prepared forward_returns.npy contract is 20 sessions")
    if config["feature_names"] != FEATURE_NAMES or config["candidate_names"] != CANDIDATE_NAMES:
        raise ValueError("Configured feature axes differ from the prepared 16-input/14-candidate contract")

    names = (*PANEL_ARRAYS, "days", "end_dates", "index_zz500")
    data = {name: np.load(folder / (name + ".npy"), mmap_mode="r", allow_pickle=False)
            for name in names}
    data["codes"] = json.loads((folder / "codes.json").read_text())
    data["features"] = np.load(expanded_features_path, mmap_mode="r", allow_pickle=False)
    days, raw_base = data["days"], data["base_eligible"]
    if days.ndim != 1 or raw_base.ndim != 2:
        raise ValueError("Prepared calendar/base eligibility axes are invalid")
    shape = raw_base.shape
    if (len(days) != shape[0] or len(data["codes"]) != shape[1]
            or data["end_dates"].shape != (shape[1],)
            or data["index_zz500"].shape != (shape[0],)):
        raise ValueError("Prepared calendar/security axes do not match")
    if (int(days[0]) != config["start_date"] or int(days[-1]) != config["end_date"]
            or np.any(np.diff(days) <= 0)):
        raise ValueError("Prepared dates differ from the requested historical calendar")
    for name in PANEL_ARRAYS:
        expected_dtype = (np.dtype("bool") if name == "base_eligible" else
                          np.dtype("uint8") if name == "universe" else
                          np.dtype("float32") if name == "forward_returns" else
                          np.dtype("float64"))
        if data[name].shape != shape or data[name].dtype != expected_dtype:
            raise ValueError(f"Prepared {name}.npy shape/dtype differs from its contract")
    if (data["features"].shape != (*shape, len(FEATURE_NAMES))
            or data["features"].dtype != np.dtype("float32")):
        raise ValueError("Expanded inputs must have shape (date, stock, 16), dtype float32")

    with Path(actions_path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(ACTION_COLUMNS):
            raise ValueError("Corporate-action CSV columns differ from the documented contract")
        actions = list(reader)
    for action in actions:
        for key in ("announcement_date", "effective_date", "listing_date", "payment_date"):
            action[key] = int(action[key]) if action[key] else None
        for key in ("ratio", "cash_price"):
            action[key] = float(action[key]) if action[key] else np.nan

    data["raw_base_eligible"] = raw_base
    data["base_eligible"] = common_eligible(raw_base, data["features"])
    data["feature_names"] = list(FEATURE_NAMES)
    data["candidate_names"] = list(CANDIDATE_NAMES)
    data["corporate_actions"] = actions
    return data
