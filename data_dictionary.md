# External inputs and calculation conventions

The fixed sample is 2018-01-02 through 2026-04-29: 2018 market sessions and 6000 historical instrument slots. Every array uses the same chronological date axis and original instrument axis, including inactive instruments. Missing numerical observations are NaN. Input data are supplied separately and are not included in this repository.

## Required files

`--data-dir` contains the following prepared files. Except where noted, matrices have shape `(2018, 6000)`.

| File | Type / meaning |
| --- | --- |
| `days.npy` | int32 `(2018,)`, ascending trading dates as YYYYMMDD |
| `codes.json` | JSON list of 6000 historical Wind-style codes or empty unused slots |
| `end_dates.npy` | int32 `(6000,)`, formal listing end dates; distant future for open listings |
| `open.npy`, `close.npy`, `preclose.npy` | float64, unadjusted CNY opening, closing and ex-reference previous closing prices |
| `adjusted_open.npy`, `adjusted_close.npy` | float64, causal synthetic price units described below |
| `upper.npy`, `lower.npy` | float64, historical daily opening price limits |
| `volume.npy`, `status.npy` | float64, traded volume and vendor trading status |
| `universe.npy` | uint8, historical universe membership |
| `industry.npy` | float64, historical first-level Shenwan industry code; NaN treated as a separate group |
| `mcap.npy` | float64, total market capitalization in ten-thousand CNY |
| `base_eligible.npy` | bool, historical base eligibility before complete-input filtering |
| `forward_returns.npy` | float32, adjusted opening return from signal t+1 to t+21; unavailable labels are NaN |
| `index_zz500.npy` | float64 `(2018,)`, CSI500 daily price-index returns, auxiliary reference |

`--features` supplies a separate float32 `(2018, 6000, 16)` prepared factor panel. Its exact column order is:

```text
F1 F2 F3 F4 F5 T1 T2 T3 T4 T5 F6 F7 F8 F9 F10 T6
```

The loader reads these prepared inputs directly. It does not need old predictions, trained weights, research-result tables or experiment metadata. Rebuilding these inputs from raw supplier financial snapshots is outside this minimal repository.

## Sources and factors

Original market inputs are vendor `KLine/k.*`, historical `Universe/uv.*`, `Industry/ind.sw.level1`, turnover `dv.s_dq_turn`, and `IndexKLine/idx.k.zz500.ret`. Financial sources are daily announcement-guarded eight-report snapshots from `ASHAREFINANCIALINDICATOR`, `ASHAREBALANCESHEET` and `ASHAREINCOME`. Both announcement dates and financial announcement dates must be finite and no later than the signal day; report periods must also be no later than that day. These guards cannot certify absence of later vendor revisions to historical snapshots.

| ID | Prepared definition |
| --- | --- |
| F1 | Four-report weighted ROE, weights 4/3/2/1 from latest to oldest, divided by its trailing 60-day mean minus one; requires at least two report values |
| F2 | 120-day change in the latest four-report ROA slope, divided by its 120-day population standard deviation |
| F3 | Same transformation as F2 applied to revenue YoY growth |
| F4 | Same transformation applied to operating-profit/parent-equity slope; retained column, unused by the fixed models and pool |
| F5 | 120-day change in the latest debt/equity ratio, divided by its 120-day population standard deviation |
| T1, T2 | Adjusted close divided by its 20- or 60-session lag, minus one |
| T3 | Trailing 20-session population standard deviation of raw close/preclose minus one |
| T4 | Adjusted close divided by its trailing 20-session mean minus one; retained column, unused |
| T5 | Trailing 20-session mean of daily turnover; invalid or negative turnover is missing |
| F6 | Latest visible parent equity divided by total market cap ×10000, retaining negative equity |
| F7 | Visible parent-profit TTM divided by total market cap ×10000, retaining losses |
| F8 | Latest visible yearly ROA (`s_fa_roa_yearly`), percentage points |
| F9 | Latest visible operating cash/revenue (`s_fa_ocftoor`) |
| F10 | Negative latest visible debt/assets (`s_fa_debttoassets`), percentage points |
| T6 | Adjusted close[t−20] / adjusted close[t−126] − 1, requiring positive finite endpoints |

For F1–F5, each eight-report row is clipped to its own mean ±3 standard deviations before transformation. Four-report slope requires four finite reports. Rolling statistics emit NaN before a full calendar window, use finite observations within it and population variance. Denominators with absolute value ≤1e−12 are missing. F1 is converted to float32 and magnitudes ≤1e−10 set to zero before use.

For F6–F10, the largest visible report period with a finite value is used; the first vendor slot resolves duplicates. Nonannual TTM profit equals current YTD + prior full year − prior same quarter, with all three reports separately visible. An annual latest report uses its full-year value. Missing reports are never replaced with later information.

Base eligibility includes historical universe membership, non-ST status, at least six months listed, ordinary Shanghai/Shenzhen A-share codes, positive volume and finite positive close, and status in 0/20/21/30/31. The common pool additionally requires finite values for all 14 candidates: F1/F2/F3/F5/T1/T2/T3/T5/F6/F7/F8/F9/F10/T6. It is identical for every method, raw baseline and equal-weight benchmark. Future label availability never affects this pool.

## Training and signals

Centered ranks use `(average_rank − 0.5) / finite_count − 0.5`, with average ties and float32 outputs. Annual feature lists and directions are fixed in `config.json`; the original four methods perform no new feature or parameter search during reproduction. Lasso repeats the fixed historical-only validation rule described below. Fundamental scores average signed financial ranks; combined scores weight the financial and technical families equally. Ridge, Lasso and LightGBM use ranked inputs and raw 20-session return labels demeaned within the eligible labeled cross section.

Training samples follow the absolute every-fifth-market-session grid. A training signal is allowed only if its t+21 exit is strictly earlier than the prediction start. The original four methods use their fixed full prior sample; the adopted Lasso uses a three-calendar-year rolling sample. Initial 2018–2020 replay uses the pre-2021 fit; annual 2021–2026 models use only mature earlier labels. Ridge and Lasso BLAS arithmetic uses one thread; LightGBM uses six threads. Fixed versions and float32 arithmetic matter for rankings.

Lasso uses the identical annual eight rank inputs as Ridge and fits float64 coefficients, with an unpenalized intercept, a complete centered float64 Gram/Xy `enet_path` with deterministic cyclic coordinate descent, `max_iter=20000` and `tol=1e-9`. Daily emitted predictions remain float32. The objective is `||y-Xw||²/(2n) + alpha*||w||1`; Ridge alpha=10000 is not on this scale. For each forecast year, the preceding calendar year supplies internal validation. Inner training rows must have t+21 exits strictly before that validation year; validation rows must have exits strictly before the forecast-year cutoff. Both use the same absolute fifth-session sampling grid.

The fixed Lasso alpha fractions are `[1e-6,1e-5,3e-5,.0001,.0003,.001,.003,.01,.03,.1,.3,.5,.7,.9,1]`. On inner training data, `alpha_max = max(abs(X.T @ (y-mean(y)))) / n`. Select the fraction with maximum equal-date mean validation Spearman Rank IC among candidates with nonzero coefficients and nonconstant cross-sectional predictions on every validation date (exact ties prefer the larger fraction). Inner training uses the three calendar years before the validation-year cutoff, while the final refit uses the three calendar years before the forecast-year cutoff; the two window lower bounds are calculated separately. Recompute alpha_max on the full mature training data and refit with the selected fraction. Fraction1 is the exact intercept-only comparison; its slopes are set to zero to eliminate numerical boundary residuals. The all-zero candidate remains a validation-loss diagnostic and is excluded from the investable Top100 selection because it cannot rank stocks. The final refit must have nonzero coefficients, and each emitted prediction date must have a nonconstant finite eligible cross section; otherwise training fails. Portfolio execution uses the same strict finite-score and Top100 rules as the other methods, without a Lasso-specific cash switch. First signal/execution are 2018-07-11/2018-07-12; earlier dates are the common preparation period. The rejected cash-policy experiment is preserved separately as a diagnostic. The adopted 2021–2026 Lasso fits retain 7/8/8/4/8/4 inputs; 2023/2025 choose the weak-penalty edge. The initial grid was expanded using only historical validation errors before inspecting portfolio returns. The already-reviewed 2026 history is not a new blind test.

The bounded optimization compared 96 Lasso and 48 fixed-Ridge configurations on 2021–2025 only: 8/14 features × expanding/3y/5y windows × demeaned-return/rank targets × MSE/RankIC inner selection × EMA0/10/20/40 (Ridge has no inner selector). Same-scale MSE is used only within its own label target; final structural comparison uses actual execution-engine fee-adjusted portfolio returns. Positive gross and net summed daily excess are required, then net summed daily excess is maximized; 1e-12 ties use positive-year count, median annual IR and lower turnover. The selected original8/3y/raw/RankIC/EMA20 configuration was frozen before its 2026 reevaluation. Development data ended 2025-12-31; the selection itself occurred on 2026-10-28, after earlier project results had already been reviewed. This is retrospective development and historical out-of-time evaluation, not a new blind test.

The paired control uses float64 Gram Ridge with fixed alpha10000 and the same selected input/window/target/EMA settings. It is not an equally tuned L2 model or the unchanged published float32 Ridge baseline. Its 2021–2025 net return is 223.12%, versus selected Lasso211.46%; original Lasso was128.45%. The selected Lasso's 2026 net3.35% is weaker than its old7.52%, so the development improvement does not establish better generalization. The grid is not reranked using2026. All necessary final settings and compact optimization evidence are in config.json; reproducing the selected strategy needs no research cache or frozen model weights.

Adopted scores first residualize prediction ranks against industry intercepts and within-industry log market cap, then rerank. EMA uses alpha=2/(span+1), with a float64 state and float32 daily emitted scores, followed by daily reranking. First finite values initialize state; missing inputs emit NaN while preserving prior state. State continues across year boundaries. The benchmark receives no predictive EMA. Returns and NAV are never smoothed.

## Prices, portfolios and costs

Adjusted prices follow a causal close/preclose chain. Adjusted open is adjusted close × raw open / raw close. This approximates reinvested corporate actions; it does not track exact cash-dividend receipts, taxes or rights subscription. Missing prices remain missing in features and labels.

Target refresh occurs every 20 market sessions on one continuous calendar; next-open trading uses the preceding close's score. Each strategy selects equal-weight Top100, while the primary benchmark equal-weights the same eligible pool. Opening limits, suspensions, T+1, cash constraints and pending claims apply to gross and net books separately. Minimum commissions, integer lots, liquidity impact and small transfer fees are omitted.

Commission is 0.0003 and slippage 0.0005 each side. Sell stamp tax is 0.001 before 2023-08-28 and 0.0005 afterward. Daily and daily doubled-slippage scopes are controls; the latter is not a 20-session cost stress.

## External corporate-action table

`--actions` supplies a CSV with these columns:

```text
old_code,new_code,announcement_date,effective_date,listing_date,ratio,cash_price,payment_date,source_url,settlement_source_url,modelling_assumption
```

Dates are YYYYMMDD. Empty optional dates become None; empty numeric values become NaN. Finite `cash_price` identifies a cash exit; otherwise the row is a share exchange. Exchange rows require a successor code, positive final ratio and verified listing date. Cash rows require positive cash price and payment date. Every row requires an announcement date no later than effectiveness and a source URL. The reported sample uses 12 sourced events; exact reproduction requires that same external event input.

Old synthetic units convert through their last valid adjusted/raw scale. Exchange claims remain unsaleable at their prior mark until successor listing, then convert using its opening scale. Cash claims revalue on payment-day close and release spendable cash next open. Certain cash payment dates use the fifth session after a sourced settlement announcement; these timing conventions belong in `modelling_assumption`. Unresolved positions after formal listing end assume zero recovery; this is a model assumption, not proof of zero economic recovery. Suspended positions otherwise carry their last valid mark, with diagnostics in the ledger.

## Outputs

Growing outputs are external. `models/` contains newly computed scores and eligibility; `results/` contains the three-scope selected-method ledger and metrics; `raw_baseline/` contains the same-pool raw-signal 20-session comparison. `report_assets/charts/` contains slide-size and report-size PNG/PDF figures; `report_assets/tables/` contains derived values, including `FourCharts_DailyValues.csv`. The three current formal documents are written to the repository root.

`models/lasso_fit_windows.csv` records annual cutoffs, selected alpha/fraction, nonzero counts, convergence, unconstrained-MSE diagnostic alpha fractions (explicitly separate from adopted RankIC selection) and rankable-candidate grid boundaries. `lasso_validation.csv` records all candidate validation losses, nonconstant validation-day counts and eligibility/selected flags and `lasso_coefficients.csv` records every annual input coefficient, including zeros. The default full run needs no previous predictions or results. `--stage lasso` adds only Lasso to existing four-method outputs and checks the repeated benchmark and original ledger subset exactly before/after merging. It preserves original score arrays.  `--stage report` consumes this pipeline's existing scores and ledgers, together with the original external inputs; it recomputes diagnostic tables and documents without model or parameter selection.

`gross_return` and `net_return` are daily portfolio returns. NAV is continuous and starts at one. `turnover` is half of total traded notional divided by opening wealth. Absolute curves use NAV minus one; excess curves sum daily strategy-minus-matching-benchmark returns in percentage points. Metrics use 252 sessions/year and a zero risk-free rate for Sharpe. Company-event counts may repeat one security across methods and books.

## V2 diagnostic conventions

Rank IC is the daily Spearman correlation between final processed scores and available mature twenty-session labels. Quintiles are assigned using all current finite eligible scores before checking future-label availability; stable security-slot order breaks score ties. Group means use the labels available inside those original groups. Missing labels never cause the remaining stocks to be reranked. Group excess subtracts that date's mean observed label in the same score pool.

Diagnostics for 2021–2025 require exits by 2025-12-31; 2026 diagnostics require exits by 2026-04-29. The final evaluable signal is 2026-03-30, leaving 55 mature signal days in 2026. These overlapping future-return cohorts are not independent observations or executable fee-adjusted portfolio returns. No ordinary independent-sample significance claim is made. Factor correlation averages defined same-date Spearman correlations on every fifth market session in 2021–2025; constant-factor pairs are omitted only where correlation is undefined.

Monthly absolute returns compound; monthly excess adds daily matched-cost differences. Rolling statistics require 252 observations and retain missing values until the window is full. Rolling volatility uses sample standard deviation times sqrt(252). Monthly turnover sums the daily single-sided values.

Monthly fee bars sum each day's cost component divided by its pre-trade opening wealth. On trading days this equals `2 * turnover * component / (buy_notional + sell_notional)`; no-trade days contribute zero. This rate sum differs from the gross/net compounded terminal-wealth gap.

Actual capital allocations replay the unchanged Ridge and benchmark net books, observing closing positions on target-refresh dates and the final date. Stock values, cash and pending claims are normalized by actual net NAV. Market-cap quintile boundaries come from that date's common eligible pool, with Q1 smallest and Q5 largest; held stocks outside that pool retain their actual values. The chart's `Other` combines unknown market capitalization, cash and claims. The replay requires every original engine value to agree exactly with the stored ledger; CSV-wide integer/float dtype widening does not change the numeric comparison.
