"""Daily, self-financing A-share simulation on causal adjusted-price units.

Scores are known at the previous close. Raw prices determine trading limits;
adjusted prices value synthetic units, approximating reinvested corporate actions.
Gross and net portfolios have separate cash and holdings. Turnover is half of
the total traded notional divided by wealth immediately before opening trades.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class _Book:
    units: np.ndarray
    target_units: np.ndarray
    cash: float = 1.0
    previous_nav: float = 1.0
    pending: dict = field(default_factory=dict)

    @property
    def pending_value(self):
        return sum(claim["mark_value"] for claim in self.pending.values())


def _prepare_actions(data):
    """Validate the small, sourced event ledger at the external-data boundary."""
    slots = {code: i for i, code in enumerate(data["codes"])}
    actions = []
    old_codes = set()
    for item in data.get("corporate_actions", []):
        action = dict(item)
        old = action["old_code"]
        if old in old_codes:
            raise ValueError(f"Duplicate corporate action for {old}")
        old_codes.add(old)
        action["old_slot"] = slots[old]
        effective = int(action["effective_date"])
        announcement = int(action["announcement_date"])
        if not 0 < announcement <= effective or not action["source_url"]:
            raise ValueError(f"Invalid corporate-action announcement/source for {old}")
        cash_price = float(action.get("cash_price", np.nan))
        if np.isfinite(cash_price):
            action["kind"] = "cash"
            action["release_date"] = int(action["payment_date"])
            if cash_price <= 0:
                raise ValueError(f"Invalid cash exit price for {old}")
        else:
            action["kind"] = "exchange"
            action["new_slot"] = slots[action["new_code"]]
            action["release_date"] = int(action["listing_date"])
            if not np.isfinite(action["ratio"]) or float(action["ratio"]) <= 0:
                raise ValueError(f"Invalid exchange ratio for {old}")
        if action["release_date"] < effective:
            raise ValueError(f"Corporate-action release precedes effectiveness for {old}")
        actions.append(action)
    return actions


def _corporate_actions(books, actions, activated, day, last_close, last_scale,
                       adjusted_open, raw_open, tradable, quote_expected, method, scope,
                       event_log):
    """Recognise dated entitlements, then release shares/cash when available."""
    counts = {"recognized": 0, "released": 0, "cash_gross": 0.0, "cash_net": 0.0}
    for event_id, action in enumerate(actions):
        old = action["old_slot"]
        if not activated[event_id] and day >= int(action["effective_date"]):
            activated[event_id] = True
            event_held = False
            for fee, book in books:
                units = book.units[old]
                if units == 0:
                    continue
                if not last_close[old] > 0 or not last_scale[old] > 0:
                    raise ValueError(f"Missing old-share conversion scale: {action['old_code']} {day}")
                raw_shares = units * last_scale[old]
                mark_value = units * last_close[old]
                book.pending[event_id] = {"raw_shares": raw_shares, "mark_value": mark_value}
                book.units[old] = 0
                book.target_units[old] = 0
                event_held = True
                event_log.append({
                    "date": int(day), "method": method, "scope": scope, "fee": fee,
                    "event": "recognition", "kind": action["kind"],
                    "old_code": action["old_code"], "new_code": action.get("new_code", ""),
                    "announcement_date": int(action["announcement_date"]),
                    "effective_date": int(action["effective_date"]),
                    "payment_date": int(action["payment_date"]) if action["kind"] == "cash" else np.nan,
                    "release_date": np.nan if action["kind"] == "cash" else action["release_date"],
                    "old_units": units, "old_adjustment_scale": last_scale[old],
                    "old_raw_shares": raw_shares, "pending_mark_value": mark_value,
                    "source_url": action["source_url"],
                })
            counts["recognized"] += int(event_held)
        if day < action["release_date"]:
            continue
        # Payment is credited after that day's market close. The earliest
        # investable cash is at the first market open strictly after payment.
        if action["kind"] == "cash" and day <= int(action["payment_date"]):
            continue
        if action["kind"] == "exchange":
            new = action["new_slot"]
            # A suspension delays availability. A normal trading day without a
            # valid opening quote is a data error, rather than an unlimited mark.
            if not tradable[new]:
                if quote_expected[new] and any(event_id in book.pending for _, book in books):
                    raise ValueError(f"Missing new-share opening quote after listing: {action['new_code']} {day}")
                continue
        released = False
        for fee, book in books:
            if event_id not in book.pending:
                continue
            claim = book.pending.pop(event_id)
            if action["kind"] == "cash":
                cash = claim["raw_shares"] * float(action["cash_price"])
                book.cash += cash
                counts[f"cash_{fee}"] += cash
                new_scale, new_units, settlement_value = np.nan, 0.0, cash
            else:
                new_scale = adjusted_open[new] / raw_open[new]
                new_units = claim["raw_shares"] * float(action["ratio"]) / new_scale
                book.units[new] += new_units
                settlement_value = new_units * adjusted_open[new]
            released = True
            event_log.append({
                "date": int(day), "method": method, "scope": scope, "fee": fee,
                "event": "settlement", "kind": action["kind"],
                "old_code": action["old_code"], "new_code": action.get("new_code", ""),
                "announcement_date": int(action["announcement_date"]),
                "effective_date": int(action["effective_date"]),
                "payment_date": int(action["payment_date"]) if action["kind"] == "cash" else np.nan,
                "release_date": int(day) if action["kind"] == "cash" else action["release_date"],
                "old_raw_shares": claim["raw_shares"],
                "ratio": action.get("ratio", np.nan),
                "cash_price": action.get("cash_price", np.nan),
                "new_adjustment_scale": new_scale, "new_units": new_units,
                "pending_mark_value": claim["mark_value"],
                "settlement_value": settlement_value,
                "settlement_mark_change": settlement_value - claim["mark_value"],
                "source_url": action["source_url"],
            })
        counts["released"] += int(released)
    return counts


def _cash_payment_close(books, actions, day, method, scope, event_log):
    """Mark paid cash at close, while retaining it outside investable opening cash."""
    changes = {"gross": 0.0, "net": 0.0}
    for event_id, action in enumerate(actions):
        if action["kind"] != "cash" or day < int(action["payment_date"]):
            continue
        for fee, book in books:
            if event_id not in book.pending:
                continue
            claim = book.pending[event_id]
            payment_value = claim["raw_shares"] * float(action["cash_price"])
            mark_change = payment_value - claim["mark_value"]
            changes[fee] += mark_change
            claim["mark_value"] = payment_value
            event_log.append({
                "date": int(day), "method": method, "scope": scope, "fee": fee,
                "event": "payment_close", "kind": "cash", "old_code": action["old_code"],
                "announcement_date": int(action["announcement_date"]),
                "effective_date": int(action["effective_date"]),
                "payment_date": int(action["payment_date"]), "release_date": np.nan,
                "old_raw_shares": claim["raw_shares"], "cash_price": action["cash_price"],
                "pending_mark_value": payment_value, "settlement_mark_change": mark_change,
                "source_url": action["source_url"],
            })
    return changes


def _target_weights(scores, eligible, top_n, benchmark):
    weights = np.zeros(len(eligible), dtype=np.float64)
    indices = np.flatnonzero(eligible)
    if benchmark:
        if len(indices):
            weights[indices] = 1.0 / len(indices)
    else:
        if not np.isfinite(scores[indices]).all():
            raise ValueError("Eligible stocks have missing or nonfinite scores")
        # Numeric slot order breaks score ties deterministically.
        order = np.lexsort((indices, -scores[indices]))
        weights[indices[order[:top_n]]] = 1.0 / top_n
    return weights


def _trade(book, marks, adjusted_open, weights, refresh, buyable, sellable,
           buy_multiplier, sell_multiplier, commission, slippage, stamp):
    opening_nav = book.cash + np.dot(book.units, marks) + book.pending_value
    if not np.isfinite(opening_nav) or opening_nav <= 0:
        raise ValueError("Nonpositive or nonfinite opening wealth")
    if refresh:
        # A suspended stock may have a close mark but no opening quote. Its
        # desired units can be set from that mark; the order cannot execute.
        book.target_units = np.divide(
            weights * opening_nav, marks, out=np.zeros_like(marks), where=marks > 0
        )
    desired_sales = np.maximum(book.units - book.target_units, 0)
    desired_buys = np.maximum(book.target_units - book.units, 0)
    sale_orders = desired_sales * marks > opening_nav * 1e-12
    buy_orders = desired_buys * marks > opening_nav * 1e-12
    blocked_sells = int(np.sum(sale_orders & ~sellable))
    blocked_buys = int(np.sum(buy_orders & ~buyable))

    # Sells happen first and are bounded by units carried into this day. No
    # stock acquired at this open can be sold again at the same open (T+1).
    sale_units = np.where(sale_orders & sellable, desired_sales, 0)
    sell_notional = float(np.dot(sale_units, np.nan_to_num(adjusted_open)))
    book.units -= sale_units
    book.cash += sell_notional * sell_multiplier

    buy_units = np.where(buy_orders & buyable, desired_buys, 0)
    buy_notional = float(np.dot(buy_units, np.nan_to_num(adjusted_open)))
    needed_cash = buy_notional * buy_multiplier
    cash_scale = min(1.0, book.cash / needed_cash) if needed_cash > 0 else 1.0
    buy_units *= cash_scale
    buy_notional *= cash_scale
    book.units += buy_units
    book.cash -= buy_notional * buy_multiplier
    if book.cash < -1e-10:
        raise ValueError("Trades created negative cash")
    book.cash = max(book.cash, 0.0)  # Only floating-point roundoff is clipped.
    fee_commission = (
        buy_notional * (1 + slippage) + sell_notional * (1 - slippage)
    ) * commission
    fee_stamp = sell_notional * (1 - slippage) * stamp
    fee_slippage = (buy_notional + sell_notional) * slippage
    return {
        "opening_nav": opening_nav,
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
        "turnover": (buy_notional + sell_notional) / (2 * opening_nav),
        "cost": fee_commission + fee_stamp + fee_slippage,
        "commission": fee_commission,
        "stamp": fee_stamp,
        "slippage": fee_slippage,
        "blocked_buys": blocked_buys,
        "blocked_sells": blocked_sells,
    }


def simulate_portfolio(data, scores, eligible, first_signal_index, config,
                       method="strategy", scope="daily", position_snapshot=None):
    """Return one gross/net portfolio pair without writing files.

    ``universe`` is used only for diagnostics at execution/valuation, never to
    choose yesterday's target. ``end_dates`` identifies actual listing ends.
    Missing suspended/no-volume quotes carry the latest mark and are counted;
    missing quotes during normal trading and unhandled listing ends fail.
    ``holding_missing_limit`` flags prolonged carries for review, rather than
    treating a long exchange suspension as a data-fetch failure.
    Sourced corporate actions recognise illiquid claims until share listing or
    cash payment. Their interim mark is the last old-share value; paid cash is
    marked at payment-day close and available at the next market open. Explicit
    ``universe_exit_policy='writeoff_enddate'`` allows zero recovery only for
    listing ends without a known exchange/cash event, separately from trades.
    An optional read-only callback observes actual closing positions on target
    refresh dates and the final date, without changing trades or valuation.
    """
    days = np.asarray(data["days"], dtype=np.int64)
    eligible = np.asarray(eligible, dtype=bool)
    n_days, n_stocks = eligible.shape
    if n_days != len(days) or np.any(np.diff(days) <= 0):
        raise ValueError("Trading dates must be strictly increasing and aligned")
    if scores is not None and np.shape(scores) != eligible.shape:
        raise ValueError("Score shape does not match eligibility")
    top_n = int(config.get("top_n", 100))
    if top_n < 1 or not 0 <= first_signal_index < n_days:
        raise ValueError("Invalid top_n or first_signal_index")
    cadence = 20 if scope == "rebalance20" else 1
    if scope not in {"daily", "rebalance20", "double_slippage"}:
        raise ValueError(f"Unknown backtest scope: {scope}")
    commission = float(config.get("commission", 0.0003))
    slippage = float(config.get("slippage", 0.0005))
    if scope == "double_slippage":
        slippage *= 2
    stamp_before = float(config.get("stamp_before", 0.001))
    stamp_after = float(config.get("stamp_after", 0.0005))
    stamp_change = int(config.get("stamp_change_date", 20230828))
    normal_status = config["tradable_status_values"]
    missing_limit = int(config.get("holding_missing_limit", 60))
    max_price_ratio = float(config.get("max_adjusted_price_ratio", 3.0))
    exit_policy = config.get("universe_exit_policy", "raise")
    if exit_policy not in {"raise", "writeoff_enddate"}:
        raise ValueError("Unrecognised universe_exit_policy")
    if min(commission, slippage, stamp_before, stamp_after) < 0:
        raise ValueError("Transaction costs cannot be negative")
    end_dates = np.asarray(data["end_dates"], dtype=np.int64)
    actions = _prepare_actions(data)
    activated = np.zeros(len(actions), dtype=bool)
    action_log = []
    last_close = np.zeros(n_stocks, dtype=np.float64)
    last_scale = np.zeros(n_stocks, dtype=np.float64)
    missing_days = np.zeros(n_stocks, dtype=np.int32)
    gross = _Book(np.zeros(n_stocks), np.zeros(n_stocks))
    net = _Book(np.zeros(n_stocks), np.zeros(n_stocks))
    target_weights = np.zeros(n_stocks, dtype=np.float64)
    records = []
    first_trade = first_signal_index + 1

    for d, day in enumerate(days):
        adjusted_open = np.asarray(data["adjusted_open"][d], dtype=np.float64)
        adjusted_close = np.asarray(data["adjusted_close"][d], dtype=np.float64)
        raw_open = np.asarray(data["open"][d], dtype=np.float64)
        raw_close = np.asarray(data["close"][d], dtype=np.float64)
        upper = np.asarray(data["upper"][d], dtype=np.float64)
        lower = np.asarray(data["lower"][d], dtype=np.float64)
        open_valid = np.isfinite(adjusted_open) & (adjusted_open > 0)
        close_valid = np.isfinite(adjusted_close) & (adjusted_close > 0)
        raw_valid = np.isfinite(raw_open) & (raw_open > 0)
        quote_expected = (data["volume"][d] > 0) & np.isin(data["status"][d], normal_status)
        tradeable = open_valid & raw_valid & quote_expected
        action_counts = _corporate_actions(
            [("gross", gross), ("net", net)], actions, activated, day,
            last_close, last_scale, adjusted_open, raw_open, tradeable,
            quote_expected, method, scope, action_log,
        )
        known_action = np.zeros(n_stocks, dtype=bool)
        converted_old = np.zeros(n_stocks, dtype=bool)
        for event_id, action in enumerate(actions):
            if int(action["announcement_date"]) <= day:
                known_action[action["old_slot"]] = True
            if activated[event_id]:
                converted_old[action["old_slot"]] = True
        held = (gross.units > 0) | (net.units > 0)
        ended = (end_dates > 0) & (end_dates < day)
        announced_wait = ended & held & known_action
        listing_ended = ended & held & ~known_action
        if np.any(listing_ended) and exit_policy == "raise":
            codes = [data["codes"][j] for j in np.flatnonzero(listing_ended)]
            raise ValueError(f"Held security passed listing EndDate on {day}: {codes}")

        # Preserve previous marks only for an explicit, counted missing quote.
        open_marks = np.where(open_valid, adjusted_open, last_close)
        previous_valid = last_close > 0
        move_check = held & previous_valid & close_valid
        price_ratio = np.divide(
            adjusted_close, last_close, out=np.ones(n_stocks), where=move_check
        )
        anomalous = move_check & (
            (price_ratio > max_price_ratio) | (price_ratio < 1 / max_price_ratio)
        )
        if np.any(anomalous):
            codes = [data["codes"][j] for j in np.flatnonzero(anomalous)]
            raise ValueError(f"Extreme adjusted-price move on {day}: {codes}")
        held_no_mark = held & (open_marks <= 0)
        if np.any(held_no_mark):
            raise ValueError(f"Held security has no opening or previous mark on {day}")
        # Missing daily limit prices cannot certify executable orders.
        buyable = tradeable & np.isfinite(upper) & (upper > 0) & (raw_open < upper)
        sellable = tradeable & np.isfinite(lower) & (lower > 0) & (raw_open > lower)
        if (ended | converted_old).any():
            buyable[ended | converted_old] = False
            sellable[ended | converted_old] = False
        refresh = d >= first_trade and (d - first_trade) % cadence == 0
        if refresh:
            null_lasso = (method == 'lasso' and
                          not np.isfinite(scores[d - 1, eligible[d - 1]]).any())
            if null_lasso:
                target_weights = np.zeros(n_stocks, dtype=np.float64)
            else:
                target_weights = _target_weights(
                    None if scores is None else scores[d - 1], eligible[d - 1],
                    top_n, method == "benchmark",
                )
            missing_target_mark = (target_weights > 0) & (open_marks <= 0)
            if missing_target_mark.any():
                codes = [data["codes"][j] for j in np.flatnonzero(missing_target_mark)]
                raise ValueError(f"Target security has no price mark on {day}: {codes}")
        stamp = stamp_before if day < stamp_change else stamp_after
        gross_trade = _trade(
            gross, open_marks, adjusted_open, target_weights, refresh,
            buyable, sellable, 1.0, 1.0, 0.0, 0.0, 0.0,
        )
        net_trade = _trade(
            net, open_marks, adjusted_open, target_weights, refresh,
            buyable, sellable, (1 + slippage) * (1 + commission),
            (1 - slippage) * (1 - commission - stamp),
            commission, slippage, stamp,
        )
        cash_mark_change = _cash_payment_close(
            [("gross", gross), ("net", net)], actions, day, method, scope, action_log,
        )

        held = (gross.units > 0) | (net.units > 0)
        missing_days = np.where(held & ~close_valid, missing_days + 1, 0)
        expected_missing_close = held & quote_expected & (
            ~close_valid | ~np.isfinite(raw_close) | (raw_close <= 0)
        )
        if expected_missing_close.any():
            codes = [data["codes"][j] for j in np.flatnonzero(expected_missing_close)]
            raise ValueError(f"Held normal-trading close quote missing on {day}: {codes}")
        close_marks = np.where(close_valid, adjusted_close, last_close)
        no_mark = held & (close_marks <= 0)
        if no_mark.any():
            raise ValueError(f"Held security has no close or previous mark on {day}")
        gross_writeoff = float(np.dot(gross.units[listing_ended], close_marks[listing_ended]))
        net_writeoff = float(np.dot(net.units[listing_ended], close_marks[listing_ended]))
        if listing_ended.any():
            for stock in np.flatnonzero(listing_ended):
                for fee, book in [("gross", gross), ("net", net)]:
                    if book.units[stock] == 0:
                        continue
                    action_log.append({
                        "date": int(day), "method": method, "scope": scope, "fee": fee,
                        "event": "conservative_writeoff", "kind": "unresolved_delisting",
                        "old_code": data["codes"][stock], "end_date": int(end_dates[stock]),
                        "old_units": book.units[stock],
                        "old_raw_shares": book.units[stock] * last_scale[stock],
                        "last_raw_price": last_close[stock] / last_scale[stock] if last_scale[stock] > 0 else np.nan,
                        "last_adjusted_price": last_close[stock],
                        "writeoff_value": book.units[stock] * close_marks[stock],
                        "source": "vendor EndDate; zero recovery modelling assumption",
                    })
            gross.units[listing_ended] = net.units[listing_ended] = 0
            gross.target_units[listing_ended] = net.target_units[listing_ended] = 0
        last_close[close_valid] = adjusted_close[close_valid]
        scale_valid = close_valid & np.isfinite(raw_close) & (raw_close > 0)
        last_scale[scale_valid] = adjusted_close[scale_valid] / raw_close[scale_valid]
        gross_nav = gross.cash + np.dot(gross.units, close_marks) + gross.pending_value
        net_nav = net.cash + np.dot(net.units, close_marks) + net.pending_value
        if not np.isfinite(gross_nav + net_nav) or min(gross_nav, net_nav) <= 0:
            raise ValueError(f"Nonpositive or nonfinite closing wealth on {day}")
        if position_snapshot is not None and d >= first_trade and (refresh or d == n_days - 1):
            position_snapshot(d, net.units, close_marks, net_nav, net.cash, net.pending_value)
        holdings_outside = held & ~np.asarray(data["universe"][d], dtype=bool)
        records.append({
            "date": int(day), "method": method, "scope": scope,
            "gross_return": gross_nav / gross.previous_nav - 1,
            "net_return": net_nav / net.previous_nav - 1,
            "gross_nav": gross_nav, "net_nav": net_nav,
            "gross_turnover": gross_trade["turnover"],
            "turnover": net_trade["turnover"],
            "gross_buy_notional": gross_trade["buy_notional"],
            "gross_sell_notional": gross_trade["sell_notional"],
            "buy_notional": net_trade["buy_notional"],
            "sell_notional": net_trade["sell_notional"],
            "transaction_cost": net_trade["cost"],
            "commission_cost": net_trade["commission"],
            "stamp_cost": net_trade["stamp"],
            "slippage_cost": net_trade["slippage"],
            "blocked_buys": net_trade["blocked_buys"],
            "blocked_sells": net_trade["blocked_sells"],
            "cash_gross": gross.cash, "cash_net": net.cash,
            "n_holdings": int(np.sum(net.units > 0)),
            "eligible_count": int(eligible[d - 1].sum()) if d >= first_trade else 0,
            "missing_valuations": int(np.sum(held & ~close_valid)),
            "missing_value_net": float(np.dot(net.units[~close_valid], close_marks[~close_valid])),
            "max_missing_days": int(missing_days.max()),
            "missing_over_limit": int(np.sum(missing_days > missing_limit)),
            "holdings_outside_universe": int(holdings_outside.sum()),
            "outside_universe_value_net": float(np.dot(net.units[holdings_outside], close_marks[holdings_outside])),
            "writeoff_count": int(listing_ended.sum()),
            "writeoff_gross": gross_writeoff, "writeoff_net": net_writeoff,
            "corporate_actions_recognized": action_counts["recognized"],
            "corporate_actions_settled": action_counts["released"],
            "corporate_cash_gross": action_counts["cash_gross"],
            "corporate_cash_net": action_counts["cash_net"],
            "corporate_cash_revaluation_gross": cash_mark_change["gross"],
            "corporate_cash_revaluation_net": cash_mark_change["net"],
            "pending_claims": len(net.pending),
            "pending_value_gross": gross.pending_value,
            "pending_value_net": net.pending_value,
            "announced_action_wait_count": int(announced_wait.sum()),
            "target_updated": refresh,
        })
        gross.previous_nav, net.previous_nav = gross_nav, net_nav
    result = pd.DataFrame.from_records(records)
    result.attrs["corporate_action_log"] = action_log
    return result


def _performance(frame, benchmark, stage, fee):
    returns = frame[f"{fee}_return"].to_numpy(dtype=np.float64)
    bench = benchmark.loc[frame["date"], f"{fee}_return"].to_numpy(dtype=np.float64)
    excess = returns - bench
    n = len(returns)
    wealth = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    vol = float(np.std(returns, ddof=1)) if n > 1 else np.nan
    exvol = float(np.std(excess, ddof=1)) if n > 1 else np.nan
    return {
        "method": frame["method"].iloc[0], "scope": frame["scope"].iloc[0],
        "stage": stage, "fee": fee, "start_date": int(frame["date"].iloc[0]),
        "end_date": int(frame["date"].iloc[-1]), "trading_days": n,
        "total_return": float(wealth[-1] - 1),
        "annualized_return": float(wealth[-1] ** (252 / n) - 1),
        "annualized_volatility": vol * np.sqrt(252),
        "zero_rf_sharpe": float(returns.mean() / vol * np.sqrt(252)) if vol > 0 else np.nan,
        "max_drawdown": float(np.min(wealth / peak - 1)),
        "cumulative_excess": float(excess.sum()),
        "annualized_mean_excess": float(excess.mean() * 252),
        "information_ratio": float(excess.mean() / exvol * np.sqrt(252)) if exvol > 0 else np.nan,
        "average_daily_turnover": float(frame["gross_turnover" if fee == "gross" else "turnover"].mean()),
        "annualized_turnover": float(frame["gross_turnover" if fee == "gross" else "turnover"].mean() * 252),
        "trading_cost_total": float(frame["transaction_cost"].sum()) if fee == "net" else 0.0,
        "blocked_buys": int(frame["blocked_buys"].sum()),
        "blocked_sells": int(frame["blocked_sells"].sum()),
        "missing_valuation_days": int((frame["missing_valuations"] > 0).sum()),
        "writeoff_count": int(frame["writeoff_count"].sum()),
        "corporate_actions_settled": int(frame["corporate_actions_settled"].sum()),
        "pending_valuation_days": int((frame["pending_claims"] > 0).sum()),
    }


def _run_group(data, scores, eligible, first_signal, scopes, output, config):
    """Run fixed score panels against their matching-cost benchmark."""
    frames, action_log = [], []
    for scope in scopes:
        for method in ["benchmark", *scores]:
            frame = simulate_portfolio(
                data, None if method == "benchmark" else scores[method],
                eligible, first_signal, config, method=method, scope=scope,
            )
            action_log.extend(frame.attrs.pop("corporate_action_log"))
            frames.append(frame)
    daily = pd.concat(frames, ignore_index=True)
    daily["stage"] = np.select(
        [daily["date"] <= int(data["days"][first_signal]),
         daily["date"] <= int(config["initial_training_end"]),
         daily["date"] < int(config["holdout_start"])],
        ["preparation", "train", "development"], default="holdout",
    )
    metrics = []
    for (scope, method), frame in daily.groupby(["scope", "method"], sort=False):
        bench = daily[(daily["scope"] == scope) & (daily["method"] == "benchmark")].set_index("date")
        for fee in ["gross", "net"]:
            daily.loc[frame.index, f"{fee}_cumulative_return"] = frame[f"{fee}_nav"].to_numpy() - 1
            excess = frame[f"{fee}_return"].to_numpy() - bench.loc[frame["date"], f"{fee}_return"].to_numpy()
            daily.loc[frame.index, f"{fee}_excess_return"] = excess
            daily.loc[frame.index, f"{fee}_cumulative_excess"] = np.cumsum(excess)
        active = frame[frame["stage"] != "preparation"]
        periods = {"full": active}
        periods.update({stage: part for stage, part in active.groupby("stage", sort=False)})
        periods.update({f"year_{year}": part for year, part in active.groupby(active["date"] // 10000, sort=False)})
        for stage, part in periods.items():
            for fee in ["gross", "net"]:
                metrics.append(_performance(part, bench, stage, fee))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "daily_returns.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(action_log).to_csv(output / "corporate_action_ledger.csv", index=False)
    return daily


def add_lasso_backtests(data, result, output_dir, config):
    """Append Lasso to existing ledgers, checking the replayed benchmark exactly."""
    output = Path(output_dir)
    experiment = output / 'lasso'
    for folder, scores, scopes in [
        ('results', result['scores'], ['daily', 'rebalance20', 'double_slippage']),
        ('raw_baseline', result['raw_scores'], ['rebalance20']),
    ]:
        _run_group(data, scores, result['eligible'], result['first_signal'], scopes,
                   experiment / folder, config)
        for filename in ['daily_returns.csv', 'metrics.csv', 'corporate_action_ledger.csv']:
            original = pd.read_csv(output / folder / filename, float_precision='round_trip')
            additional = pd.read_csv(experiment / folder / filename, float_precision='round_trip')
            if filename == 'daily_returns.csv':
                left = original[original.method.eq('benchmark')].reset_index(drop=True)
                right = additional[additional.method.eq('benchmark')].reset_index(drop=True)
                pd.testing.assert_frame_equal(left, right, check_exact=True)
            merged = pd.concat([original[~original.method.eq('lasso')],
                                additional[additional.method.eq('lasso')]], ignore_index=True)
            merged.to_csv(output / folder / filename, index=False)
            restored = pd.read_csv(output / folder / filename, float_precision='round_trip')
            pd.testing.assert_frame_equal(
                original[~original.method.eq('lasso')].reset_index(drop=True),
                restored[~restored.method.eq('lasso')].reset_index(drop=True), check_exact=True)


def run_backtests(data, result, output_dir, config):
    """Reproduce the fixed 20-session strategy, raw baseline and daily controls.

    Daily and double-slippage controls retain daily target refresh. The main and
    raw baseline refresh every twenty sessions from the original first trade.
    All variants use daily valuation, identical eligibility and matched costs.
    No training, feature selection or parameter search occurs in this module.
    """
    output = Path(output_dir)
    first_signal = int(result["first_signal"])
    daily = _run_group(data, result["scores"], result["eligible"], first_signal,
                       ["daily", "rebalance20", "double_slippage"], output / "results", config)
    _run_group(data, result["raw_scores"], result["eligible"], first_signal,
               ["rebalance20"], output / "raw_baseline", config)
    return daily
