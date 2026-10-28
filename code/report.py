"""V2 diagnostics, publication figures, English report and editable slides.

Consumes only the fixed strategy inputs and its current model/portfolio outputs.
No model selection, score change or return smoothing is performed here.
"""


"""Read-only numerical analysis for the fixed Group 9 reporting pipeline."""

from pathlib import Path as _analysis_Path
import json as _analysis_json
import pickle as _analysis_pickle

import numpy as _analysis_np
import pandas as _analysis_pd
from scipy.stats import rankdata as _analysis_rankdata


_analysis_methods = ('fundamental', 'combined', 'ridge', 'lasso', 'lightgbm')
_analysis_all_methods = (*_analysis_methods, 'benchmark')
_MAIN_MODEL_VERSIONS = {'ridge': 'Ridge-exp', 'lasso': 'Lasso-3y'}
_MAIN_PERIODS = ('full', 'development', 'holdout')


def _main_comparison(ctx):
    """One ledger-backed comparison shared by report, slides and CSV export."""
    metrics = ctx['metrics']
    rows = []
    for period in _MAIN_PERIODS:
        for method, model_version in _MAIN_MODEL_VERSIONS.items():
            selected = metrics.loc[metrics.scope.eq('rebalance20')
                                   & metrics.stage.eq(period) & metrics.method.eq(method)]
            net = selected.loc[selected.fee.eq('net')].iloc[0]
            gross = selected.loc[selected.fee.eq('gross')].iloc[0]
            rows.append({'period': period, 'method': method, 'model_version': model_version,
                         'net_return': float(net.total_return),
                         'gross_return': float(gross.total_return),
                         'net_excess_pp': float(net.cumulative_excess * 100),
                         'gross_excess_pp': float(gross.cumulative_excess * 100)})
    return _analysis_pd.DataFrame(rows)


def _analysis_calendar(series, years):
    pd, np = _analysis_pd, _analysis_np
    result = pd.DataFrame(np.nan, index=pd.Index(years, name='year'),
                          columns=pd.Index(range(1, 13), name='month'))
    for date, value in series.items():
        result.loc[date.year, date.month] = value
    return result


def _analysis_rank_ic(score, label):
    np = _analysis_np
    left = _analysis_rankdata(score, method='average').astype(np.float64)
    right = _analysis_rankdata(label, method='average').astype(np.float64)
    left -= left.mean()
    right -= right.mean()
    norm = np.sqrt(np.sum(left * left) * np.sum(right * right))
    return float(np.sum(left * right) / norm) if norm > 0 else np.nan


def _analysis_factor_corr(data, features):
    np, pd = _analysis_np, _analysis_pd
    columns = [data['feature_names'].index(name) for name in features]
    size = len(columns)
    total, counts = np.zeros((size, size)), np.zeros((size, size), dtype=np.int64)
    sampled = 0
    for day in np.flatnonzero((data['days'] >= 20210101) & (data['days'] <= 20251231)):
        if day % 5:
            continue
        valid = data['base_eligible'][day]
        values = np.asarray(data['features'][day, valid][:, columns], dtype=np.float64)
        if len(values) < 30:
            continue
        ranks = _analysis_rankdata(values, method='average', axis=0)
        ranks -= ranks.mean(axis=0)
        norm = np.sqrt(np.sum(ranks * ranks, axis=0))
        denominator = np.outer(norm, norm)
        correlation = np.full((size, size), np.nan)
        np.divide(ranks.T @ ranks, denominator, out=correlation, where=denominator > 0)
        finite = np.isfinite(correlation)
        total[finite] += correlation[finite]
        counts[finite] += 1
        sampled += 1
    mean = np.full((size, size), np.nan)
    np.divide(total, counts, out=mean, where=counts > 0)
    return pd.DataFrame(mean, index=features, columns=features), sampled


def _analysis_label_diagnostics(data, scores, eligible):
    np, pd = _analysis_np, _analysis_pd
    days = np.asarray(data['days'])
    daily, grouped = [], {}
    for day in range(len(days) - 21):
        signal, exit_day = int(days[day]), int(days[day + 21])
        if signal < 20210101:
            period, cutoff = '2018–2020', 20201231
        elif signal < 20260101:
            period, cutoff = '2021–2025', 20251231
        else:
            period, cutoff = '2026', int(days[-1])
        if exit_day > cutoff:
            continue
        labels = np.asarray(data['forward_returns'][day], dtype=np.float64)
        for method in _analysis_methods:
            current = eligible[day] & np.isfinite(scores[method][day])
            slots = np.flatnonzero(current)
            if not len(slots):
                continue
            available = np.isfinite(labels[slots])
            count = int(available.sum())
            ic = (_analysis_rank_ic(scores[method][day, slots[available]], labels[slots[available]])
                  if count >= 30 else np.nan)
            daily.append({'date': pd.to_datetime(str(signal)), 'exit_date': pd.to_datetime(str(exit_day)),
                          'year': signal // 10000, 'method': method, 'rank_ic': ic, 'n': count,
                          'mature_period': period})
            if period == '2018–2020':
                continue
            # Group assignment uses every currently observable score. Label
            # availability affects observations inside an existing group only.
            order = np.lexsort((slots, np.asarray(scores[method][day, slots])))
            ordered_slots = slots[order]
            groups = np.minimum(np.arange(len(slots)) * 5 // len(slots), 4) + 1
            benchmark = float(labels[slots[available]].mean()) if count else np.nan
            for group in range(1, 6):
                members = ordered_slots[groups == group]
                observed = labels[members]
                observed = observed[np.isfinite(observed)]
                key = (period, method, group)
                record = grouped.setdefault(key, {'return_sum': 0., 'excess_sum': 0., 'days': 0,
                                                   'labelled_stock_days': 0, 'assigned_stock_days': 0})
                record['assigned_stock_days'] += len(members)
                record['labelled_stock_days'] += len(observed)
                if len(observed):
                    value = float(observed.mean())
                    record['return_sum'] += value
                    record['excess_sum'] += value - benchmark
                    record['days'] += 1
    ic_daily = pd.DataFrame(daily)
    ic_annual = ic_daily.groupby(['year', 'method'], sort=True).agg(
        mean_ic=('rank_ic', 'mean'), days=('rank_ic', 'count'), mean_n=('n', 'mean'),
        last_signal=('date', 'max')).reset_index()
    # Retain every model-year explicitly, including missing diagnostic observations.
    model_years = pd.MultiIndex.from_product(
        [np.unique(days // 10000), _analysis_methods], names=['year', 'method'])
    ic_annual = ic_annual.set_index(['year', 'method']).reindex(model_years).reset_index()
    ic_annual['days'] = ic_annual['days'].fillna(0).astype(int)
    ic_annual['last_signal'] = ic_annual.last_signal.dt.strftime('%Y-%m-%d')
    quintiles = []
    for (period, method, group), values in grouped.items():
        count = values['days']
        quintiles.append({'period': period, 'method': method, 'group': group,
                          'mean_return': values['return_sum'] / count if count else np.nan,
                          'mean_excess': values['excess_sum'] / count if count else np.nan,
                          'days': count,
                          'coverage': values['labelled_stock_days'] / values['assigned_stock_days'],
                          'labelled_stock_days': values['labelled_stock_days'],
                          'assigned_stock_days': values['assigned_stock_days']})
    return ic_daily, ic_annual, pd.DataFrame(quintiles)


def analyze_results(data, config, output_root):
    """Analyze fixed outputs without refitting, selecting, or changing returns."""
    np, pd = _analysis_np, _analysis_pd
    output = _analysis_Path(output_root)
    tables = output / 'report_assets/tables'
    tables.mkdir(parents=True, exist_ok=True)
    assets = tables.parent
    metrics = pd.read_csv(output / 'results/metrics.csv', float_precision='round_trip')
    raw_metrics = pd.read_csv(output / 'raw_baseline/metrics.csv', float_precision='round_trip')
    # Preserve the V1 figure pipeline's CSV-to-curve arithmetic exactly.
    daily = pd.read_csv(output / 'results/daily_returns.csv')
    raw_daily = pd.read_csv(output / 'raw_baseline/daily_returns.csv')
    daily['date'] = pd.to_datetime(daily.date.astype(str))
    raw_daily['date'] = pd.to_datetime(raw_daily.date.astype(str))
    daily_main = daily[daily.scope.eq('rebalance20')].sort_values(['date', 'method']).reset_index(drop=True)
    raw_daily = raw_daily[raw_daily.scope.eq('rebalance20')].sort_values(['date', 'method']).reset_index(drop=True)
    returns = {fee: daily_main.pivot(index='date', columns='method', values=f'{fee}_return').reindex(columns=_analysis_all_methods)
               for fee in ('gross', 'net')}
    raw_returns = {fee: raw_daily.pivot(index='date', columns='method', values=f'{fee}_return').reindex(columns=_analysis_all_methods)
                   for fee in ('gross', 'net')}
    net_wealth = (1 + returns['net']).cumprod()
    drawdown = net_wealth / net_wealth.cummax().clip(lower=1.) - 1
    annual_returns = {fee: (1 + frame).groupby(frame.index.year).prod() - 1 for fee, frame in returns.items()}
    annual_excess = {fee: frame.subtract(frame.benchmark, axis=0).groupby(frame.index.year).sum() * 100
                     for fee, frame in returns.items()}
    month = pd.offsets.MonthEnd()
    excess = returns['net'].ridge - returns['net'].benchmark
    monthly_net_excess = (excess.resample(month).sum() * 100).rename('ridge')
    monthly_net_return = ((1 + returns['net'].ridge).resample(month).prod() - 1).rename('ridge')
    years = range(returns['net'].index[0].year, returns['net'].index[-1].year + 1)
    calendar_excess = _analysis_calendar(monthly_net_excess, years)
    calendar_return = _analysis_calendar(monthly_net_return, years)
    rolling_excess = returns['net'].subtract(returns['net'].benchmark, axis=0).rolling(252, min_periods=252).sum() * 100
    rolling_vol = returns['net'][['ridge', 'benchmark']].rolling(252, min_periods=252).std(ddof=1) * np.sqrt(252) * 100
    turnover = daily_main.pivot(index='date', columns='method', values='turnover')[['ridge', 'benchmark']]
    monthly_turnover = turnover.resample(month).sum() * 100
    ridge = daily_main[daily_main.method.eq('ridge')].set_index('date')
    notionals = (ridge.buy_notional + ridge.sell_notional).to_numpy(dtype=np.float64)
    daily_costs = pd.DataFrame(index=ridge.index)
    for name in ('commission', 'stamp', 'slippage'):
        values = np.zeros(len(ridge), dtype=np.float64)
        np.divide(2 * ridge.turnover.to_numpy() * ridge[f'{name}_cost'].to_numpy(), notionals,
                  out=values, where=notionals > 0)
        daily_costs[name] = values * 100
    cost_components = daily_costs.resample(month).sum()
    dates = pd.to_datetime(np.asarray(data['days']).astype(str))
    coverage = pd.DataFrame({'date': dates, 'base_count': data['raw_base_eligible'].sum(axis=1),
                             'common_count': data['base_eligible'].sum(axis=1)})
    adopted = {name for spec in config['method_specs'].values() for name in spec['features']}
    selected_features = [name for name in data['feature_names'] if name in adopted]
    factor_corr, corr_days = _analysis_factor_corr(data, selected_features)
    scores = {method: np.load(output / 'models' / f'scores_{method}.npy', mmap_mode='r') for method in _analysis_methods}
    eligible = np.load(output / 'models/eligible.npy', mmap_mode='r')
    ic_daily, ic_annual, quintiles = _analysis_label_diagnostics(data, scores, eligible)
    first = int(np.flatnonzero(eligible.sum(axis=1) >= config['models']['first_signal_min_stocks'])[0])
    last_label = len(dates) - 22
    summary = {
        'start': dates[0].strftime('%Y-%m-%d'), 'end': dates[-1].strftime('%Y-%m-%d'),
        'days': len(dates), 'stock_slots': eligible.shape[1], 'slots': eligible.shape[1],
        'first_signal': dates[first].strftime('%Y-%m-%d'), 'first_trade': dates[first + 1].strftime('%Y-%m-%d'),
        'first_signal_index': first, 'first_trade_index': first + 1,
        'last_label_signal': dates[last_label].strftime('%Y-%m-%d'),
        'label_last_exit': dates[-1].strftime('%Y-%m-%d'), 'mature_label_days': len(dates) - 21,
        'labelled_2026_days': int(ic_daily[ic_daily.year.eq(2026)].date.nunique()),
        'latest_year': dates[-1].year, 'common_pool_min': int(coverage.common_count.min()),
        'common_pool_max': int(coverage.common_count.max()),
        'mean_base_count': float(coverage.base_count.mean()), 'mean_common_count': float(coverage.common_count.mean()),
        'active_common_pool_min': int(coverage.common_count.iloc[first:].min()),
        'active_common_pool_mean': float(coverage.common_count.iloc[first:].mean()),
        'selected_features': selected_features, 'factor_corr_sample_days': corr_days,
    }
    summary.update(start_date=summary['start'], end_date=summary['end'], label_last_signal=summary['last_label_signal'])
    notes = [
        '2018–2020 model scores are a fitted replay. Later models use strictly mature historical training labels; this report does not refit or choose a strategy.',
        'Label diagnostics are grouped by signal date. Replay labels must exit by 2020-12-31; 2021–2025 labels must exit by 2025-12-31; 2026 labels must exit by the data endpoint. Annual IC rows group the retained observations by signal year.',
        'IC is daily Spearman correlation between the final processed score and available mature 20-session forward-return labels. Its stock count is the finite paired cross section.',
        'Quintiles are formed from all currently finite eligible scores before checking forward-label availability: group 1 is lowest and group 5 highest. Fixed security-slot order breaks exact score ties; group sizes differ by at most one stock.',
        'Quintile means average daily formed cohorts equally. Excess subtracts that date’s same-score-pool mean observed label. Missing labels do not change the original group assignment; coverage is labelled/assigned stock-days.',
        'Forward-20-session cohort returns are label diagnostics, not executable fee-adjusted portfolio alpha. Overlapping labels are not independent observations; no conventional independent-sample significance claim is made.',
        'Factor correlation is the equal-date mean of defined cross-sectional Spearman correlations on the common pool, sampled at full-calendar index multiples of five during 2021–2025. Constant-factor dates have undefined correlations and are omitted for that pair.',
        'Monthly costs sum daily commission/stamp/slippage divided by the net book’s pre-trade opening NAV. The identity is component/opening_NAV = 2*turnover*component/(buy_notional+sell_notional); no-trade days contribute zero. This sum is not the gross-minus-net compounded terminal-wealth gap.',
        'Monthly turnover sums daily single-sided turnover; rolling excess sums daily strategy-minus-benchmark returns over 252 sessions. Rolling volatility uses sample standard deviation and sqrt(252). Preparation cash days remain in their actual calendar positions.',
        'Main and raw-baseline books share the original offset-zero 20-session execution grid and the same cost and trading restrictions. Daily/double-slippage diagnostics retain their recorded separate scopes.',
    ]
    lasso_fits = pd.read_csv(output / 'models/lasso_fit_windows.csv')
    lasso_coefficients = pd.read_csv(output / 'models/lasso_coefficients.csv')
    lasso_validation = pd.read_csv(output / 'models/lasso_validation.csv')
    ctx = {'lasso_fits': lasso_fits, 'lasso_coefficients': lasso_coefficients,
           'lasso_validation': lasso_validation,
           'main_comparison': _main_comparison({'metrics': metrics}), 'summary': summary, 'config': config, 'metrics': metrics, 'raw_metrics': raw_metrics,
           'daily_main': daily_main, 'raw_daily': raw_daily, 'returns': returns, 'raw_returns': raw_returns,
           'drawdown': drawdown, 'annual_returns': annual_returns, 'annual_excess': annual_excess,
           'monthly_net_excess': monthly_net_excess, 'monthly_net_return': monthly_net_return,
           'calendar_excess': calendar_excess, 'calendar_return': calendar_return,
           'rolling_excess': rolling_excess, 'rolling_vol': rolling_vol, 'monthly_turnover': monthly_turnover,
           'cost_components': cost_components, 'coverage': coverage, 'factor_corr': factor_corr,
           'ic_annual': ic_annual, 'ic_daily': ic_daily, 'quintiles': quintiles, 'diagnostics_notes': notes}
    exported = {'metrics': metrics, 'raw_metrics': raw_metrics, 'coverage': coverage,
                'factor_corr': factor_corr, 'ic_annual': ic_annual, 'ic_daily': ic_daily, 'quintiles': quintiles,
                'monthly_net_excess': monthly_net_excess, 'monthly_net_return': monthly_net_return,
                'calendar_excess': calendar_excess, 'calendar_return': calendar_return,
                'rolling_excess': rolling_excess, 'rolling_vol': rolling_vol,
                'monthly_turnover': monthly_turnover, 'cost_components': cost_components,
                'lasso_fit_windows': lasso_fits, 'lasso_coefficients': lasso_coefficients,
                'lasso_validation': lasso_validation, 'main_comparison': ctx['main_comparison']}
    for fee in ('gross', 'net'):
        exported.update({f'annual_returns_{fee}': annual_returns[fee], f'annual_excess_{fee}': annual_excess[fee],
                         f'returns_{fee}': returns[fee], f'raw_returns_{fee}': raw_returns[fee]})
    for name, frame in exported.items():
        frame.to_csv(tables / f'{name}.csv', index=not isinstance(frame.index, pd.RangeIndex))
    (tables / 'summary.json').write_text(_analysis_json.dumps(summary, indent=2) + '\n')
    (tables / 'diagnostics_notes.json').write_text(_analysis_json.dumps(notes, indent=2, ensure_ascii=False) + '\n')
    with (assets / 'context.pkl').open('wb') as stream:
        _analysis_pickle.dump(ctx, stream, protocol=4)
    return ctx


from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import PercentFormatter, MaxNLocator
from matplotlib import dates as mdates
import matplotlib
from backtest import simulate_portfolio


_CHART_METHODS=['fundamental','combined','ridge','lasso','lightgbm','benchmark']
_CHART_LABELS={'fundamental':'Fund.','combined':'Combined','ridge':_MAIN_MODEL_VERSIONS['ridge'],'lasso':_MAIN_MODEL_VERSIONS['lasso'],'lightgbm':'LGBM','benchmark':'EW pool'}
_CHART_COLORS={'fundamental':'#9063B4','combined':'#D79538','ridge':'#137F82','lasso':'#C45262','lightgbm':'#3475AB','benchmark':'#8795A6'}
_CHART_INK='#162B42'
_CHART_MUTED='#657489'
_CHART_GRID='#E3E9EF'


def _position_diagnostics(data,config,output_root):
    """Replay two unchanged books and observe actual marked net allocations."""
    output=Path(output_root)
    eligible=np.load(output/'models/eligible.npy',mmap_mode='r')
    first=int(np.flatnonzero(eligible.sum(axis=1)>=100)[0])
    original=pd.read_csv(output/'results/daily_returns.csv',float_precision='round_trip')
    records=[]
    for method in ['ridge','benchmark']:
        def observe(day,units,marks,nav,cash,claims):
            weights=units*marks/nav
            cap=np.asarray(data['mcap'][day])
            pool=eligible[day]&np.isfinite(cap)&(cap>0)
            boundaries=np.quantile(cap[pool],[.2,.4,.6,.8])
            known=np.isfinite(cap)&(cap>0)
            groups=np.searchsorted(boundaries,cap,side='right')
            values=[float(weights[known&(groups==i)].sum()) for i in range(5)]
            buckets=['Q1 (small)','Q2','Q3','Q4','Q5 (large)']
            values.extend([float(weights[~known].sum()),cash/nav,claims/nav])
            buckets.extend(['Unknown','Cash','Claims'])
            np.testing.assert_allclose(sum(values),1,rtol=0,atol=1e-12)
            date=pd.to_datetime(str(int(data['days'][day])))
            for bucket,value in zip(buckets,values):
                records.append({'date':date,'method':method,'category':'cap','bucket':bucket,'weight':value})
            industries=np.asarray(data['industry'][day])
            for group in np.unique(industries[np.isfinite(industries)&(weights>0)]):
                records.append({'date':date,'method':method,'category':'industry',
                                'bucket':f'SW1 {int(group)}','weight':float(weights[industries==group].sum())})
            records.append({'date':date,'method':method,'category':'industry','bucket':'Unknown',
                            'weight':float(weights[~np.isfinite(industries)].sum())})
        scores=None if method=='benchmark' else np.load(output/'models/scores_ridge.npy',mmap_mode='r')
        frame=simulate_portfolio(data,scores,eligible,first,config,method=method,scope='rebalance20',position_snapshot=observe)
        expected=original[original.scope.eq('rebalance20')&original.method.eq(method)].reset_index(drop=True)
        # CSV concatenation widens all-zero integer columns when another book
        # contains fractional claims; require exact values across that boundary.
        pd.testing.assert_frame_equal(frame,expected[frame.columns],check_exact=True,check_dtype=False)
    positions=pd.DataFrame(records)
    tables=output/'report_assets/tables'
    tables.mkdir(parents=True,exist_ok=True)
    positions.to_csv(tables/'actual_position_allocations.csv',index=False)
    (tables/'position_replay_checks.json').write_text(json.dumps({'scope':'rebalance20','methods':['ridge','benchmark'],
        'engine_columns':len(frame.columns),'unchanged_ledger':True,'actual_closing_weights':True,'allocation_rows':len(positions)},indent=2)+'\n')
    return positions


def _chart_metric(ctx,scope,stage,method,fee='net',raw=False):
    table=ctx['raw_metrics'] if raw else ctx['metrics']
    return table[table.scope.eq(scope)&table.stage.eq(stage)&table.method.eq(method)&table.fee.eq(fee)].iloc[0]


def _make_chart(ctx, key, path, width, height, size):
    """Render each chart at its final size, with physical rather than fractional margins."""
    fig = Figure(figsize=(width, height), facecolor='white')
    FigureCanvasAgg(fig)
    report = size == 12
    left, right = (.90, .16) if report else (1.00, .22)
    bottom, top = (.44, .72) if report else (.50, .75)
    ax = fig.add_axes([left / width, bottom / height,
                       (width-left-right) / width, (height-bottom-top) / height])
    legend_columns = None
    time_axis = False
    percent_axis = False
    multi_axes = False
    plot_order = ['fundamental', 'combined', 'lightgbm', 'benchmark', 'ridge', 'lasso']
    if key in ('gross_absolute', 'net_absolute', 'gross_excess', 'net_excess'):
        fee = key.split('_')[0]
        r = ctx['returns'][fee]
        excess = key.endswith('excess')
        curves = r.sub(r.benchmark, axis=0).cumsum()*100 if excess else ((1+r).cumprod()-1)*100
        for method in plot_order:
            if excess and method == 'benchmark':
                continue
            ax.plot(curves.index, curves[method], color=_CHART_COLORS[method],
                    label=_CHART_LABELS[method], linewidth=(2.2 if report else 2.8) if method == 'ridge' else (1.05 if report else 1.5),
                    linestyle=(0, (4, 2)) if method == 'benchmark' else '-',
                    alpha=1 if method == 'ridge' else .9, zorder=5 if method == 'ridge' else 3)
        if excess:
            ax.axhline(0, color=_CHART_COLORS['benchmark'], lw=1, label='EW pool (0)')
            ax.set_ylabel('Excess (pp)', fontsize=size, color=_CHART_MUTED, labelpad=8)
            both = [v.sub(v.benchmark, axis=0).cumsum().to_numpy()*100 for v in ctx['returns'].values()]
        else:
            percent_axis = True
            both = [((1+v).cumprod().to_numpy()-1)*100 for v in ctx['returns'].values()]
        lo, hi = min(np.nanmin(v) for v in both), max(np.nanmax(v) for v in both)
        ax.set_ylim(min(0, lo)-.05*(hi-lo), hi+.07*(hi-lo))
        ax.scatter(curves.index[-1], curves.ridge.iloc[-1], s=23 if report else 35,
                   color=_CHART_COLORS['ridge'], zorder=6, clip_on=False)
        legend_columns = 3
        time_axis = True
    elif key == 'drawdown':
        frame = ctx['drawdown']*100
        ax.fill_between(frame.index, frame.ridge.to_numpy(), 0,
                        color=_CHART_COLORS['ridge'], alpha=.10, linewidth=0)
        for method in ('ridge', 'benchmark'):
            ax.plot(frame.index, frame[method], label=_CHART_LABELS[method],
                    color=_CHART_COLORS[method], lw=1.7 if report else 2.1,
                    ls=(0, (4, 2)) if method == 'benchmark' else '-')
        ax.set_ylim(-38, 1)
        ax.set_yticks([-30, -20, -10, 0])
        legend_columns, time_axis, percent_axis = 2, True, True
    elif key == 'risk_return':
        ax.set_position([left/width, .78/height, (width-left-right)/width, (height-.78-top)/height])
        offsets = {'fundamental': (10, -18), 'combined': (-72, 12), 'ridge': (-20, 14),
                   'lasso': (10, -19), 'lightgbm': (10, 5), 'benchmark': (-20, -19)}
        points = []
        for method in _CHART_METHODS:
            row = _chart_metric(ctx, 'rebalance20', 'development', method)
            x, y = row.annualized_volatility*100, row.annualized_return*100
            points.append((x, y))
            ax.scatter(x, y, s=115 if method == 'ridge' else 70, color=_CHART_COLORS[method],
                       edgecolor='white', linewidth=1, zorder=5)
            ax.annotate(_CHART_LABELS[method], (x, y), xytext=offsets[method],
                        textcoords='offset points', color=_CHART_COLORS[method], fontsize=size,
                        fontweight='bold' if method == 'ridge' else 'normal')
        xy = np.asarray(points)
        ax.set_xlim(xy[:, 0].min()-1.0, xy[:, 0].max()+1.25)
        ax.set_ylim(xy[:, 1].min()-4, xy[:, 1].max()+4.5)
        ax.set_xlabel('Annualized volatility (%)', color=_CHART_MUTED, fontsize=size, labelpad=8)
        ax.set_ylabel('Annualized net return (%)', color=_CHART_MUTED, fontsize=size, labelpad=8)
        ax.grid(axis='x', color=_CHART_GRID, linewidth=.65)
    elif key in ('calendar_excess', 'calendar_return', 'factor_corr'):
        fig.delaxes(ax)
        corr = key == 'factor_corr'
        table = ctx['factor_corr'] if corr else ctx['calendar_excess'] if key == 'calendar_excess' else ctx['calendar_return']*100
        values = table.to_numpy(dtype=float)
        cmap = LinearSegmentedColormap.from_list('group9_balanced', ['#BC6655', '#F8FAFB', '#146C80'])
        cmap.set_bad('white' if corr else '#F1F4F6')
        bound = 1 if corr else max(1, float(np.ceil(np.nanmax(np.abs(values)))))
        if corr:
            values = np.where(np.tril(np.ones_like(values, dtype=bool)), values, np.nan)
            side = min(width-.78, height-1.03)
            ax = fig.add_axes([((width-side)/2+.05)/width, .60/height, side/width, side/height])
        else:
            ax = fig.add_axes([.79/width, .65/height, (width-.95)/width, (height-1.15)/height])
        im = ax.imshow(values, cmap=cmap, norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
                       aspect='equal' if corr else 'auto', interpolation='nearest')
        ax.set_xticks(np.arange(values.shape[1]), list(table.columns) if corr else
                      ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.set_yticks(np.arange(values.shape[0]), list(table.index) if corr else
                      [str(y)+('*' if y == 2026 else '') for y in table.index])
        ax.xaxis.tick_top()
        ax.tick_params(axis='both', length=0, pad=6, labelsize=size, colors=_CHART_INK)
        if not corr:
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    v = values[i, j]
                    label = '—' if np.isnan(v) else f'{v:+.1f}' if abs(v) >= .05 else '0.0'
                    color = '#9AA6B2' if np.isnan(v) else 'white' if abs(v) > bound*.62 else _CHART_INK
                    ax.text(j, i, label, ha='center', va='center', fontsize=size, color=color)
        ax.set_xticks(np.arange(-.5, values.shape[1]), minor=True)
        ax.set_yticks(np.arange(-.5, values.shape[0]), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.2)
        ax.tick_params(which='minor', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        color_width = 1.6 if report else 2.2
        color_x = .70 if report else .95
        cax = fig.add_axes([color_x/width, .36/height, color_width/width, .09/height])
        bar = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-bound, 0, bound])
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=size, length=0, pad=3, colors=_CHART_MUTED)
        unit = 'Spearman correlation' if corr else 'Monthly excess (pp)' if key == 'calendar_excess' else 'Monthly return (%)'
        fig.text((color_x+color_width+.32)/width, .31/height, unit, fontsize=size, color=_CHART_MUTED)
        multi_axes = True
    elif key == 'annual_return':
        frame = ctx['annual_returns']['net']
        x = np.arange(len(frame))
        ax.bar(x-.19, frame.ridge*100, .34, label=_CHART_LABELS['ridge'], color=_CHART_COLORS['ridge'], zorder=3)
        ax.bar(x+.19, frame.benchmark*100, .34, label='EW pool', color='#A2AEBA', zorder=3)
        years_to_label = list(range(0,len(frame),2)) if report else list(range(len(frame)))
        ax.set_xticks(x[years_to_label], [str(frame.index[i])+('*' if frame.index[i] == 2026 else '') for i in years_to_label])
        ax.axhline(0, color='#ACB7C1', lw=.8)
        percent_axis, legend_columns = True, 2
    elif key in ('rolling_excess', 'rolling_vol', 'turnover', 'costs', 'coverage'):
        if key == 'rolling_excess':
            frame = ctx['rolling_excess'].ridge
            ax.plot(frame.index, frame, color=_CHART_COLORS['ridge'], lw=2)
            ax.axhline(0, color='#9BA9B5', lw=.8)
            ax.set_ylabel('Excess (pp)', fontsize=size, color=_CHART_MUTED, labelpad=8)
            time_axis = True
        elif key in ('rolling_vol', 'turnover'):
            frame = ctx['rolling_vol'] if key == 'rolling_vol' else ctx['monthly_turnover']
            for method in ('ridge', 'benchmark'):
                ax.plot(frame.index, frame[method], label=_CHART_LABELS[method],
                        color=_CHART_COLORS[method], lw=1.6 if report else 2,
                        ls=(0, (4, 2)) if method == 'benchmark' else '-')
            percent_axis, legend_columns, time_axis = True, 2, True
            if key == 'turnover':
                ax.set_ylim(bottom=0)
        elif key == 'costs':
            frame = ctx['cost_components'].groupby(ctx['cost_components'].index.year).sum()
            x = np.arange(len(frame))
            base = np.zeros(len(frame))
            for col, color, label in [('commission', '#8BA7B6', 'Commission'),
                                      ('stamp', '#D3A452', 'Stamp tax'),
                                      ('slippage', '#146C80', 'Slippage')]:
                ax.bar(x, frame[col], width=.60, bottom=base, color=color, label=label, zorder=3)
                base += frame[col].to_numpy()
            years_to_label = list(range(0,len(frame),2)) if report else list(range(len(frame)))
            ax.set_xticks(x[years_to_label], [str(frame.index[i])+('*' if frame.index[i] == 2026 else '') for i in years_to_label])
            ax.set_ylabel('Annual cost (pp)', fontsize=size, color=_CHART_MUTED, labelpad=8)
            ax.set_ylim(0, base.max()*1.16)
            legend_columns = 3
        else:
            frame = ctx['coverage']
            for col, label, method in [('base_count', 'Historical base pool', 'benchmark'),
                                       ('common_count', 'Complete-input pool', 'ridge')]:
                ax.plot(frame.date, frame[col], label=label, color=_CHART_COLORS[method],
                        lw=1.6 if report else 2.2, ls=(0,(4,2)) if method == 'benchmark' else '-')
            ax.set_ylim(0, np.ceil(frame.base_count.max()/1000)*1000+100)
            ax.set_ylabel('Stocks', fontsize=size, color=_CHART_MUTED, labelpad=8)
            legend_columns, time_axis = 2, True
    elif key == 'rank_ic':
        table = ctx['ic_annual'].query('year >= 2021').pivot(index='year', columns='method', values='mean_ic')
        x = np.arange(len(table))
        for k, method in enumerate(_CHART_METHODS[:-1]):
            ax.bar(x+(k-2)*.16, table[method], width=.14,
                   label=_CHART_LABELS[method], color=_CHART_COLORS[method], zorder=3)
        ax.set_xticks(x, [str(y)+('*' if y == 2026 else '') for y in table.index])
        ax.set_ylabel('Mean daily Rank IC', fontsize=size, color=_CHART_MUTED, labelpad=8)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.02))
        ax.axhline(0, color='#ACB7C1', lw=.8)
        legend_columns = 3
    elif key in ('quintile', 'quintile_2026'):
        period = '2021–2025' if key == 'quintile' else '2026'
        table = ctx['quintiles'].query('period == @period')
        for method in ['fundamental', 'combined', 'lightgbm', 'ridge', 'lasso']:
            rows = table.query('method == @method').sort_values('group')
            ax.plot(rows.group, rows.mean_excess*100, marker='o', markersize=5 if report else 7,
                    color=_CHART_COLORS[method], label=_CHART_LABELS[method],
                    lw=2 if method == 'ridge' else 1.3, zorder=5 if method == 'ridge' else 3)
        values = ctx['quintiles'].mean_excess*100
        span = values.max()-values.min()
        ax.set_ylim(values.min()-.12*span, values.max()+.15*span)
        ax.set_xlim(.75, 5.25)
        ax.set_xticks(range(1,6), ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax.set_ylabel('Mean 20-day excess (pp)', fontsize=size, color=_CHART_MUTED, labelpad=8)
        ax.axhline(0, color='#9AA8B5', lw=.8)
        legend_columns = 3
    elif key == 'controls':
        fig.delaxes(ax)
        if report:
            panel_height = (height-1.90)/2
            ax1 = fig.add_axes([.75/width, (height-.45-panel_height)/height, (width-.95)/width, panel_height/height])
            ax2 = fig.add_axes([.75/width, .65/height, (width-.95)/width, panel_height/height])
        else:
            ax1 = fig.add_axes([.70/width, .70/height, (width/2-1.10)/width, (height-1.35)/height])
            ax2 = fig.add_axes([(.50*width+.55)/width, .70/height, (width/2-.80)/width, (height-1.35)/height])
        x = np.arange(len(_CHART_METHODS) - 1)
        for offset, raw, label, color in [(-.18, True, 'Raw', '#A4AFBB'), (.18, False, 'Adopted', _CHART_COLORS['ridge'])]:
            vals = [_chart_metric(ctx, 'rebalance20', 'development', m, raw=raw).total_return*100 for m in _CHART_METHODS[:-1]]
            ax1.bar(x+offset, vals, .32, label=label, color=color, zorder=3)
        ax1.set_xticks(x, [_CHART_LABELS[m] for m in _CHART_METHODS[:-1]])
        ax1.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax1.set_title('2021–2025 net return', loc='left', color=_CHART_INK, fontsize=size, pad=10)
        ax1.legend(loc='upper left', frameon=False, fontsize=size, ncol=2, handlelength=1, columnspacing=1)
        ax1.set_ylim(top=max(p.get_height() for p in ax1.patches)*1.25)
        vals = [_chart_metric(ctx, scope, 'holdout', 'ridge').cumulative_excess*100
                for scope in ('daily', 'rebalance20', 'double_slippage')]
        ax2.bar(np.arange(3), vals, .55, color=['#A4AFBB', _CHART_COLORS['ridge'], '#C9D0D7'], zorder=3)
        ax2.set_xticks(range(3), ['Daily', 'Every 20', 'Daily\n2× slippage'])
        ax2.axhline(0, color='#A4AFBB', lw=.8)
        ax2.set_title('2026 net excess (pp)', loc='left', color=_CHART_INK, fontsize=size, pad=10)
        for child in (ax1, ax2):
            child.grid(axis='y', color=_CHART_GRID, lw=.7)
            child.set_axisbelow(True)
            child.yaxis.set_major_locator(MaxNLocator(4))
            child.tick_params(axis='both', labelsize=size, length=0, pad=5, colors=_CHART_MUTED)
            child.spines[['top', 'right', 'left']].set_visible(False)
            child.spines['bottom'].set_color(_CHART_GRID)
        multi_axes = True
    elif key == 'cap_weights':
        table = ctx['positions']
        table = table[table.date.eq(table.date.max()) & table.category.eq('cap')]
        frame = table.pivot(index='bucket', columns='method', values='weight').reindex(
            ['Q1 (small)', 'Q2', 'Q3', 'Q4', 'Q5 (large)', 'Unknown', 'Cash', 'Claims'])
        frame.loc['Other'] = frame.loc[['Unknown', 'Cash', 'Claims']].sum()
        frame = frame.iloc[[0,1,2,3,4,8]]
        ax.set_position([1.00/width, .43/height, (width-1.25)/width, (height-.96)/height])
        y = np.arange(6)
        ax.barh(y-.17, frame.ridge*100, .30, color=_CHART_COLORS['ridge'], label=_CHART_LABELS['ridge'], zorder=3)
        ax.barh(y+.17, frame.benchmark*100, .30, color='#A4AFBB', label='EW pool', zorder=3)
        ax.set_yticks(y, ['Small Q1', 'Q2', 'Q3', 'Q4', 'Large Q5', 'Other'])
        ax.invert_yaxis()
        ax.set_xlim(0, 32)
        ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.xaxis.set_major_locator(MaxNLocator(4))
        legend_columns = 2
    else:
        raise KeyError(key)
    if not multi_axes:
        axis = 'x' if key == 'cap_weights' else 'y'
        ax.grid(axis=axis, color=_CHART_GRID, linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#CFD8E0')
        ax.tick_params(axis='both', labelsize=size, length=0, pad=6, colors=_CHART_MUTED)
        if key not in ('drawdown', 'cap_weights', 'rank_ic'):
            ax.yaxis.set_major_locator(MaxNLocator(4 if height < 3.3 else 5))
        if percent_axis:
            ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        if time_axis:
            ax.set_xlim(pd.Timestamp(ctx['summary']['start']), pd.Timestamp(ctx['summary']['end']))
            years = list(range(2019, 2027)) if width > 9 else [2020, 2022, 2024, 2026]
            ax.set_xticks(pd.to_datetime([ctx['summary']['start']]+[f'{year}-01-01' for year in years]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        if legend_columns:
            handles, labels = ax.get_legend_handles_labels()
            if key in ('gross_absolute', 'net_absolute', 'gross_excess', 'net_excess', 'rank_ic', 'quintile', 'quintile_2026'):
                legend_order = [_CHART_LABELS[m] for m in _CHART_METHODS] + ['EW pool (0)']
                order = sorted(range(len(labels)), key=lambda i: legend_order.index(labels[i]))
                handles, labels = [handles[i] for i in order], [labels[i] for i in order]
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.53, .995),
                       ncol=legend_columns, frameon=False, fontsize=size,
                       handlelength=1.35, handletextpad=.45, columnspacing=.8)
    path = Path(path)
    fig.savefig(path.with_suffix('.pdf'), metadata={'Title': f'Group 9 | {key}'})
    fig.savefig(path.with_suffix('.png'), dpi=220)
    return {'pdf': str(path.with_suffix('.pdf')), 'png': str(path.with_suffix('.png')),
            'width': width, 'height': height}


def make_charts(ctx, output_root):
    """Separate physical geometry for A4 pages, full-slide charts and square matrices."""
    root = Path(output_root)/'report_assets'
    folder = root/'charts'
    folder.mkdir(parents=True, exist_ok=True)
    rw = 595.275590551/72-2
    dimensions = {
        'gross_absolute': ((10.6,4.2),(rw,3.0)),
        'net_absolute': ((10.6,4.2),(rw,3.0)),
        'gross_excess': ((10.6,4.2),(rw,3.0)),
        'net_excess': ((10.6,4.2),(rw,3.0)),
        'drawdown': ((5.2,4.1),(rw,3.5)),
        'risk_return': ((5.2,4.1),(rw,4.4)),
        'calendar_excess': ((8.2,4.5),(rw,3.6)),
        'annual_return': ((10.6,4.2),(rw,2.8)),
        'rolling_excess': ((5.2,4.1),(rw,3.0)),
        'rolling_vol': ((5.2,4.1),(rw,3.0)),
        'turnover': ((10.6,4.2),(rw,3.0)),
        'costs': ((10.6,4.2),(rw,3.0)),
        'coverage': ((10.6,4.2),(rw,3.2)),
        'factor_corr': ((6.4,5.0),(rw,5.5)),
        'rank_ic': ((10.6,4.2),(rw,4.2)),
        'quintile': ((10.6,4.2),(rw,3.0)),
        'quintile_2026': ((10.6,4.2),(rw,3.0)),
        'controls': ((10.6,4.2),(rw,4.6)),
        'cap_weights': ((8.0,4.2),(rw,3.2)),
    }
    charts = {}
    with matplotlib.rc_context({'font.family':'DejaVu Sans', 'pdf.fonttype':42,
                                'axes.unicode_minus':False, 'figure.facecolor':'white'}):
        for key, pair in dimensions.items():
            charts[key] = {}
            for mode, (width,height) in zip(('slide','report'),pair):
                charts[key][mode] = _make_chart(ctx,key,folder/f'{key}_{mode}',width,height,12 if mode=='report' else 14)
    rows = []
    for method in _CHART_METHODS:
        gross, net = ctx['returns']['gross'], ctx['returns']['net']
        rows.append(pd.DataFrame({'date':gross.index,'method':method,
            'gross_absolute':(1+gross[method]).cumprod().to_numpy()-1,
            'net_absolute':(1+net[method]).cumprod().to_numpy()-1,
            'gross_excess_pp':(gross[method]-gross.benchmark).cumsum().to_numpy()*100,
            'net_excess_pp':(net[method]-net.benchmark).cumsum().to_numpy()*100}))
    pd.concat(rows,ignore_index=True).to_csv(root/'tables/FourCharts_DailyValues.csv',index=False,float_format='%.12g')
    ctx['cost_components'].groupby(ctx['cost_components'].index.year).sum().to_csv(root/'tables/costs_annual.csv')
    (root/'chart_manifest.json').write_text(json.dumps(charts,indent=2)+'\n')
    return charts

from pathlib import Path
from html import escape as _report_escape

import fitz as _report_fitz
import pandas as _report_pd
from reportlab.pdfgen.canvas import Canvas as _report_Canvas
from reportlab.lib.pagesizes import A4 as _report_A4
from reportlab.lib.styles import ParagraphStyle as _report_ParagraphStyle
from reportlab.lib import colors as _report_colors
from reportlab.platypus import Paragraph as _report_Paragraph
from reportlab.platypus import Table as _report_Table
from reportlab.platypus import TableStyle as _report_TableStyle


_report_ink = '#162B42'
_report_teal = '#137F82'
_report_muted = '#657489'
_report_grid = '#E3E9EF'
_report_methods = ['fundamental', 'combined', 'ridge', 'lasso', 'lightgbm', 'benchmark']
_report_names = {
    'fundamental': 'Fundamental', 'combined': 'Combined', 'ridge': _MAIN_MODEL_VERSIONS['ridge'],
    'lasso': _MAIN_MODEL_VERSIONS['lasso'], 'lightgbm': 'LightGBM', 'benchmark': 'Benchmark',
}


def _report_metric(ctx, method, stage='full', fee='net', scope='rebalance20', raw=False):
    frame = ctx['raw_metrics'] if raw else ctx['metrics']
    return frame.loc[
        frame['method'].eq(method) & frame['stage'].eq(stage)
        & frame['fee'].eq(fee) & frame['scope'].eq(scope)
    ].iloc[0]


def _report_percent(value, digits=1):
    return f'{float(value):.{digits}%}'


def _report_pp(value, digits=2):
    return f'{float(value) * 100:+.{digits}f}'


def _report_date(value):
    return _report_pd.Timestamp(value).strftime('%d %b %Y').lstrip('0')


def _report_paragraph(text, kind='body'):
    return ('p', kind, text)


def _report_chart(key, caption):
    return [('chart', key), _report_paragraph(caption, 'caption'), ('gap', 8)]


def _report_table(rows, widths):
    return ('table', rows, widths)


def _report_render_table(rows, widths, styles):
    cells = [
        [_report_Paragraph(_report_escape(str(value)), styles['table_header' if i == 0 else 'table'])
         for value in row]
        for i, row in enumerate(rows)
    ]
    table = _report_Table(cells, colWidths=widths, hAlign='LEFT')
    table.setStyle(_report_TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), _report_colors.HexColor('#E7F0F2')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LINEBELOW', (0, 0), (-1, 0), .6, _report_colors.HexColor(_report_teal)),
        ('LINEBELOW', (0, -1), (-1, -1), .4, _report_colors.HexColor(_report_grid)),
    ]))
    return table


def _lasso_report_settings(config):
    """Describe the frozen Lasso specification without research-file dependencies."""
    spec = config['method_specs']['lasso']
    params = spec['model_params']
    window = params['training_window_years']
    return {
        'features': len(spec['features']),
        'window': 'expanding history' if window is None else f'{window}-year rolling history',
        'target': 'demeaned return target' if params['target_transform'] == 'raw' else 'cross-sectional rank target',
        'selector': 'mean daily MSE' if params['selection_metric'] == 'mean_daily_mse' else 'mean daily Rank IC',
        'selector_action': 'minimizes' if params['selection_metric'] == 'mean_daily_mse' else 'maximizes',
        'span': int(spec['processing']['prediction_rank_ema_span']),
        'alpha_count': len(params['alpha_fractions']),
    }


def write_report(ctx, charts, project, output_root):
    """Write fifteen independently laid-out A4 body pages and references.

    Charts are embedded from their vector PDFs at their native report size.
    Values come from the current fixed-strategy ledgers and diagnostics in ctx.
    """
    project, output_root = Path(project), Path(output_root)
    config, summary = ctx['config'], ctx['summary']
    page_w, page_h = _report_A4
    width = page_w - 144
    styles = {
        'body': _report_ParagraphStyle(
            'ReportBody', fontName='Times-Roman', fontSize=12, leading=18,
            textColor=_report_colors.HexColor(_report_ink)),
        'caption': _report_ParagraphStyle(
            'ReportCaption', fontName='Times-Roman', fontSize=12, leading=18,
            textColor=_report_colors.HexColor(_report_ink)),
        'heading': _report_ParagraphStyle(
            'ReportHeading', fontName='Times-Bold', fontSize=17, leading=22,
            textColor=_report_colors.HexColor(_report_ink)),
        'subheading': _report_ParagraphStyle(
            'ReportSubheading', fontName='Times-Bold', fontSize=12, leading=18,
            textColor=_report_colors.HexColor(_report_teal)),
        'table': _report_ParagraphStyle(
            'ReportTable', fontName='Times-Roman', fontSize=12, leading=18,
            textColor=_report_colors.HexColor(_report_ink)),
        'table_header': _report_ParagraphStyle(
            'ReportTableHeader', fontName='Times-Bold', fontSize=12, leading=18,
            textColor=_report_colors.HexColor(_report_ink)),
    }
    net = {method: _report_metric(ctx, method) for method in _report_methods}
    gross = {method: _report_metric(ctx, method, fee='gross') for method in _report_methods}
    recent = {method: _report_metric(ctx, method, 'holdout') for method in _report_methods}
    recent_gross = {method: _report_metric(ctx, method, 'holdout', 'gross') for method in _report_methods}
    development = {method: _report_metric(ctx, method, 'development') for method in _report_methods}
    ridge, benchmark = net['ridge'], net['benchmark']
    lasso_differences = [development['lasso'].total_return - development['ridge'].total_return,
                         recent['lasso'].total_return - recent['ridge'].total_return]
    lasso_comparison = ('Lasso exceeds original Ridge in both intervals.' if min(lasso_differences) > 0
                        else 'Lasso trails original Ridge in both intervals.' if max(lasso_differences) < 0
                        else 'Lasso improves on original Ridge in development, but trails it in 2026.'
                        if lasso_differences[0] > 0 and lasso_differences[1] < 0
                        else 'The Lasso–original Ridge comparison varies across intervals.')
    end_date = _report_date(summary['end_date'])
    first_trade = _report_date(summary['first_trade'])
    label_last = _report_date(summary['label_last_signal'])
    specs = config['method_specs']
    ridge_params = specs['ridge']['model_params']
    tree_params = specs['lightgbm']['model_params']
    model_settings = config['models']
    lasso_settings = _lasso_report_settings(config)
    optimization = config.get('lasso_optimization')

    comparison = _main_comparison(ctx)
    main_rows = [['Model version', 'Full net', '2021–2025 net', '2026 net']]
    for version in _MAIN_MODEL_VERSIONS.values():
        values = comparison.loc[comparison.model_version.eq(version)].set_index('period')
        main_rows.append([version, *[_report_percent(values.loc[period, 'net_return'], 2)
                                    for period in _MAIN_PERIODS]])

    # Page 1: the same published versions and three periods as the headline CSV.
    summary_rows = [['Portfolio', '2026 gross', '2026 net', 'Net excess (pp)']]
    for method in _report_methods:
        summary_rows.append([
            _report_names[method], _report_percent(recent_gross[method]['total_return'], 2),
            _report_percent(recent[method]['total_return'], 2),
            _report_pp(recent[method]['cumulative_excess']),
        ])
    pages = [[
        _report_paragraph('Fundamental and technical signals for A-share stock selection', 'heading'),
        ('gap', 10), _report_paragraph('Investments 1005 | Group 9', 'subheading'),
        _report_paragraph(
            f'Historical daily data: {_report_date(summary["start_date"])} to {end_date}. '
            f'Long-only equal-weight Top-{config["top_n"]} targets refresh every '
            f'{config["rebalance_sessions"]} market sessions, with the same pool, execution and costs.'),
        ('gap', 8), _report_paragraph('One published comparison, three periods', 'subheading'),
        _report_paragraph(
            'Ridge-exp is the original expanding-history Ridge. Lasso-3y is the optimized '
            'three-year-window Lasso. These exact versions appear in the main curves and '
            'tables. The full history includes 2018–2020 fitted replay; 2021–2025 is '
            'development, and 2026 is the reviewed partial-year reevaluation.'), ('gap', 8),
        _report_table(main_rows, [102, 106, 132, width - 340]), ('gap', 8),
        _report_paragraph(
            'Lasso-3y has the higher full-history and development net return; Ridge-exp '
            'has the higher 2026 net return. These are different intervals of the same '
            'published models. Ridge-3y control is a separate development diagnostic, '
            'not the Ridge-exp main curve (Section 13).'), ('gap', 8),
        _report_paragraph(f'All methods: 2026 through {end_date}', 'subheading'),
        _report_table(summary_rows, [94, 99, 99, width - 292]), ('gap', 8),
        _report_paragraph(
            'The research question is whether accounting and market signals improve '
            'stock selection after trading costs within a common opportunity set. '
            'Absolute returns compound daily wealth; excess sums daily matched-benchmark '
            'return differences. Higher development or fitted-history returns do not '
            'establish stronger future performance.'),
    ]]

    # Page 2: concise definitions preserve the economic meaning of every candidate.
    factor_rows = [
        ['ID', 'Prepared input and investment hypothesis'],
        ['F1', 'Four-report weighted ROE relative to its trailing 60-day mean: changing profitability.'],
        ['F2', '120-day change in the four-report ROA slope / trailing 120-day standard deviation.'],
        ['F3', 'The same slope-change transformation applied to revenue YoY growth.'],
        ['F5', '120-day change in debt/equity / trailing 120-day standard deviation: financial risk.'],
        ['F6', 'Latest visible parent equity / total market value, retaining negative equity.'],
        ['F7', 'Visible parent-profit TTM / total market value, retaining losses.'],
        ['F8', 'Latest visible yearly ROA level: profitability.'],
        ['F9', 'Latest visible operating cash flow / revenue: cash generation.'],
        ['F10', 'Negative latest visible debt/assets: lower leverage.'],
        ['T1, T2', 'Adjusted-close returns over 20 and 60 sessions: trend or reversal.'],
        ['T3', 'Trailing 20-session population standard deviation of daily returns: volatility.'],
        ['T5', 'Trailing 20-session mean turnover: trading activity and liquidity.'],
        ['T6', 'Adjusted close at t-20 / adjusted close at t-126 minus one: medium-term momentum.'],
    ]
    pages.append([
        _report_paragraph('1. Economic rationale and factor definitions', 'heading'), ('gap', 10),
        _report_paragraph(
            'Profitability, growth, valuation and leverage may help explain relative returns. '
            'Price trends, volatility and turnover may capture information diffusion, reversal or '
            'liquidity. These are economic hypotheses; a plausible explanation alone does not '
            'establish a profitable trading rule.'), ('gap', 8),
        _report_table(factor_rows, [52, width - 52]), ('gap', 10),
        _report_paragraph(
            f'The adopted annual models use the union {", ".join(summary["selected_features"])}. '
            'The 14 candidates above also define the common complete-input stock pool. '
            'F10 and T6 need not enter an adopted forecast to affect that common pool. '
            'Factor directions and annual input lists are fixed in config.json; the full '
            'formulas, report selection and missing-value conventions are in data_dictionary.md.'),
        ('gap', 8),
        _report_paragraph(
            'Financial inputs require report periods and both recorded announcement dates '
            'no later than the signal day. This prevents using known later report dates, '
            'but cannot certify that vendor snapshots preserve the original unrevised values.'),
    ])

    # Page 3: coverage is a count of available names, never a holdings claim.
    coverage_rows = [
        ['Data characteristic', 'Recorded value'],
        ['Market sessions / historical slots', f'{int(summary["days"]):,} / {int(summary["slots"]):,}'],
        ['Mean base-eligible names', f'{float(summary["mean_base_count"]):,.0f}'],
        ['Mean common complete-input names', f'{float(summary["mean_common_count"]):,.0f}'],
        ['First signal / first execution', f'{_report_date(summary["first_signal"])} / {first_trade}'],
    ]
    pages.append([
        _report_paragraph('2. Historical data and the common opportunity set', 'heading'), ('gap', 10),
        _report_paragraph(
            'The prepared daily panel aligns market prices, volume, turnover, historical universe '
            'membership, ST status, listing age, industry, market capitalization and financial '
            'snapshots on a fixed historical security axis. Inactive and subsequently delisted '
            'instruments remain on that axis; membership is not determined by survival today.'),
        ('gap', 8), _report_table(coverage_rows, [242, width - 242]), ('gap', 8),
        *_report_chart('coverage',
            'Figure 1. Daily base-eligible and common complete-input stock counts. '
            'These are available names, not actual portfolio holdings. Table means '
            'include every recorded session and the initial zero complete-input counts.'),
        _report_paragraph(
            'Eligibility requires historical Shanghai/Shenzhen A-share membership, non-ST '
            'status, at least six months listed, positive traded volume and valid market inputs. '
            'The common pool additionally requires all 14 candidate values. Every model, raw '
            'baseline and benchmark uses this same pool; future-label availability does not '
            'determine whether a stock can be selected. The benchmark equal-weights this '
            'pool with the same cadence, execution limits and matched costs; it is not '
            'the CSI500 price index.'), ('gap', 8),
        _report_paragraph(
            'Complete-input filtering favors firms with a longer and more complete reporting '
            'history. Initial cash while lagged features accumulate explains the flat '
            f'beginning before {first_trade}.'),
    ])

    # Page 4: the one explicit fitted-history statement and actual fixed settings.
    model_rows = [['Method', 'Construction and fixed settings']]
    for method in _report_methods[:-1]:
        spec = specs[method]
        span = int(spec['processing']['prediction_rank_ema_span'])
        if method == 'fundamental':
            construction = 'Mean signed financial ranks.'
        elif method == 'combined':
            construction = 'Financial and technical family scores receive equal weight.'
        elif method == 'ridge':
            construction = f'L2-regularized regression; alpha={ridge_params["alpha"]:g}. [1]'
        elif method == 'lasso':
            construction = f'L1 regression; {lasso_settings["features"]} inputs; annual validation alpha. [5]'
        else:
            construction = (
                f'Depth {tree_params["max_depth"]}; {tree_params["num_leaves"]} leaves; '
                f'{tree_params["boosting_rounds"]} trees; minimum leaf '
                f'{tree_params["min_child_samples"]}; learning rate '
                f'{model_settings["learning_rate"]:g}. [2]')
        model_rows.append([
            _report_names[method], f'{construction} Score EMA span {span}.',
        ])
    pages.append([
        _report_paragraph('3. Models and implementable portfolios', 'heading'), ('gap', 10),
        _report_paragraph(
            'Each daily cross section is converted to centered average ranks. The prediction '
            f'target is the adjusted opening return from t+1 to t+{config["horizon"] + 1}, '
            'demeaned across eligible labeled stocks; a configured rank target transforms '
            'this label cross-sectionally for Lasso.'), ('gap', 8),
        _report_table(model_rows, [88, width - 88]), ('gap', 10),
        _report_paragraph(
            'Scores are residualized against industry and within-industry log size, reranked, '
            'and causally smoothed with EMA alpha=2/(span+1), then reranked again. State '
            'continues across years; missing inputs emit no score. Returns and NAV are not smoothed.'),
        ('gap', 8),
        _report_paragraph(
            f'Annual refits sample prior history every '
            f'{model_settings["training_stride"]} market sessions. A training observation is '
            'admitted only when its t+21 exit is strictly earlier than the prediction start. '
            'The 2018–2020 portion is a fitted replay using the pre-2021 model; from 2021, '
            'annual models use mature earlier labels. Original models use expanding history; '
            'Lasso uses its configured input set, history window and target (Section 13).' ),
        ('gap', 8),
        _report_paragraph('Execution and costs', 'subheading'),
        _report_paragraph(
            'Top-100 equal-weight targets refresh on one continuous 20-session calendar '
            'and trade at the next open using preceding-close scores. Valuation and '
            'blocked-order retries remain daily. Suspensions, price limits, cash and T+1 '
            'can prevent targets being reached. Gross and net books are simulated separately '
            'and compared with benchmarks under their respective fee assumptions.'), ('gap', 8),
        _report_paragraph(
            f'Commission is {config["commission"] * 10000:g} bp and slippage '
            f'{config["slippage"] * 10000:g} bp per side. Sell stamp duty is '
            f'{config["stamp_before"] * 10000:g} bp before 28 Aug 2023 and '
            f'{config["stamp_after"] * 10000:g} bp thereafter. [3, 4] Round lots, '
            'minimum commissions and liquidity impact are omitted.'), ('gap', 6),
        _report_paragraph(
            'The causal adjusted-price chain approximates reinvested corporate actions. '
            'Sourced exchanges and cash exits follow settlement assumptions; quote-less '
            'suspensions retain their last valid mark. Formal unresolved listing ends '
            'assume zero recovery rather than proving zero economic value.'),
    ])

    # Pages 5–6 reserve their space for the four main curves.
    pages.append([
        _report_paragraph('4. Complete-history absolute returns', 'heading'), ('gap', 10),
        _report_paragraph(
            'The complete-history curves compare Ridge-exp and Lasso-3y alongside all '
            'other methods. NAV minus one includes 2018–2020 fitted replay, cash, execution '
            'constraints and daily compounding.'), ('gap', 8),
        *_report_chart('gross_absolute',
            f'Figure 2. Cumulative gross returns, 2 Jan 2018–{end_date}. '
            'Each strategy is compared with the gross common-pool benchmark.'),
        *_report_chart('net_absolute',
            f'Figure 3. Cumulative net returns over the same complete history, including '
            'commission, sell stamp duty and slippage.'),
        _report_paragraph(
            f'Lasso-3y ends at {_report_percent(gross["lasso"]["total_return"], 2)} gross '
            f'and {_report_percent(net["lasso"]["total_return"], 2)} net; Ridge-exp ends at '
            f'{_report_percent(gross["ridge"]["total_return"], 2)} gross and '
            f'{_report_percent(ridge["total_return"], 2)} net. These are the full-history '
            'figures in the first-page comparison. Gross-minus-net terminal wealth includes '
            'compounding and differing trading paths, not simply fee payments.'),
    ])
    pages.append([
        _report_paragraph('5. Complete-history excess returns', 'heading'), ('gap', 10),
        _report_paragraph(
            'Excess sums daily strategy returns minus the matching benchmark, in '
            'percentage points. The benchmark is zero by construction. This is not compounded '
            'relative wealth.'), ('gap', 8),
        *_report_chart('gross_excess',
            'Figure 4. Cumulative daily gross excess versus the same-pool gross benchmark, '
            f'2 Jan 2018–{end_date}.'),
        *_report_chart('net_excess',
            'Figure 5. Cumulative daily net excess versus the same-pool net benchmark, '
            'including the matching execution costs.'),
        _report_paragraph(
            f'Full-history net excess is {_report_pp(net["lasso"]["cumulative_excess"])} pp '
            f'for Lasso-3y and {_report_pp(ridge["cumulative_excess"])} pp for Ridge-exp. '
            f'In 2021–2025 it is {_report_pp(development["lasso"]["cumulative_excess"])} '
            f'and {_report_pp(development["ridge"]["cumulative_excess"])} pp; in 2026 '
            f'through {end_date}, {_report_pp(recent["lasso"]["cumulative_excess"])} '
            f'and {_report_pp(recent["ridge"]["cumulative_excess"])} pp, respectively. '
            'Falling segments show benchmark outperformance; the model versions stay fixed.'),
    ])

    # Page 8: return is paired with observed risk, not a one-number sales claim.
    risk_rows = [['Portfolio', 'Net return', 'Net CAGR', 'Ann. vol.', 'Max. DD']]
    for method in _report_methods:
        row = net[method]
        risk_rows.append([
            _report_names[method], _report_percent(row['total_return']), _report_percent(row['annualized_return']),
            _report_percent(row['annualized_volatility']), _report_percent(row['max_drawdown']),
        ])
    pages.append([
        _report_paragraph('6. Drawdowns and complete-history risk', 'heading'), ('gap', 10),
        *_report_chart('drawdown',
            'Figure 6. Net-wealth drawdowns relative to each portfolio’s prior peak. '
            'Negative values measure losses from that peak.'),
        _report_table(risk_rows, [91, 88, 88, 87, width - 354]), ('gap', 8),
        _report_paragraph(
            f'Ridge’s worst drawdown is {_report_percent(abs(ridge["max_drawdown"]))}, '
            f'slightly deeper than the benchmark’s '
            f'{_report_percent(abs(benchmark["max_drawdown"]))}. Its annualized volatility '
            f'is {_report_percent(ridge["annualized_volatility"])}, compared with '
            f'{_report_percent(benchmark["annualized_volatility"])}. The historical return '
            'advantage does not remove market exposure, drawdowns or model risk. '
            'Annualization uses 252 market sessions; no risk-free return is assumed for Sharpe.'),
    ])

    pages.append([
        _report_paragraph('7. The return–risk trade-off in 2021–2025', 'heading'), ('gap', 10),
        _report_paragraph(
            'The same 2021–2025 interval is used for every method. A higher point '
            'has a higher compounded annual growth rate; a point farther right '
            'has more annualized daily volatility.'), ('gap', 8),
        *_report_chart('risk_return',
            'Figure 7. Net CAGR versus annualized daily volatility in 2021–2025, '
            'with identical 20-session target schedules and matched costs.'),
        _report_paragraph(
            f'Lasso-3y has net CAGR {_report_percent(development["lasso"]["annualized_return"], 2)} '
            f'and volatility {_report_percent(development["lasso"]["annualized_volatility"], 2)}; '
            f'Ridge-exp has {_report_percent(development["ridge"]["annualized_return"], 2)} '
            f'and {_report_percent(development["ridge"]["annualized_volatility"], 2)}, respectively. '
            'These are the same published versions as the full-history curves, evaluated '
            'only on 2021–2025. Ridge-3y control is auxiliary and is not plotted here.'), ('gap', 8),
        _report_paragraph(
            'The development return–risk advantage of Lasso-3y does not carry into '
            'the 2026 net-return comparison. Fitted history, development and partial-year '
            'reevaluation answer different questions and must retain their stated dates.'),
    ])

    # Page 9: a different aggregation is explicitly distinguished from compounding.
    pages.append([
        _report_paragraph('8. Performance across months and calendar years', 'heading'), ('gap', 10),
        _report_paragraph(
            'Monthly excess sums daily return differences; annual returns compound '
            'wealth. Both expose variation hidden by the final cumulative return.'), ('gap', 8),
        *_report_chart('calendar_excess',
            'Figure 8. Ridge-exp monthly net excess in percentage points: the sum of daily '
            'strategy-minus-benchmark returns. Gray dashes denote unobserved months.'),
        *_report_chart('annual_return',
            f'Figure 9. Compounded calendar-year net returns. The 2026 observations '
            f'end on {end_date}; the final bar is a partial year.'),
        _report_paragraph(
            f'The 2021–2025 Ridge net return is '
            f'{_report_percent(development["ridge"]["total_return"])}, while 2026 through '
            f'{end_date} returns {_report_percent(recent["ridge"]["total_return"], 2)}. '
            'Negative monthly excess occurs within the profitable history. The partial '
            '2026 bar is not directly comparable with full-year bars.'),
    ])

    # Page 10: fee bars use their own accounting denominator, not NAV drag.
    pages.append([
        _report_paragraph('9. Turnover and the sources of trading costs', 'heading'), ('gap', 10),
        *_report_chart('turnover',
            'Figure 10. Monthly summed daily turnover for Ridge-exp and the benchmark. '
            'Daily turnover is half the traded notional divided by opening wealth.'),
        *_report_chart('costs',
            'Figure 11. Calendar-year Ridge-exp commission, stamp duty and slippage, '
            'scaled by each transaction day’s pre-trade opening NAV and summed (pp). '
            '2026 ends on 29 Apr; the 2023 tax change applies for part of that year.'),
        _report_paragraph(
            f'Mean daily turnover is {_report_percent(ridge["average_daily_turnover"], 2)} '
            f'for Ridge, versus {_report_percent(benchmark["average_daily_turnover"], 2)} '
            'for the benchmark. Blocked-order retries can create trades between target '
            'refreshes. The cost bars are accounting charges, not the gross-minus-net '
            'terminal return difference.'),
    ])

    # Page 11: factor dependence and forecast quality, without conflating either with P&L.
    pages.append([
        _report_paragraph('10. Dependence among the adopted inputs', 'heading'), ('gap', 10),
        _report_paragraph(
            'Daily cross-sectional rank correlations reveal which inputs carry overlapping '
            'information. They describe statistical dependence rather than causality '
            'or profitable trading after costs.'), ('gap', 8),
        *_report_chart('factor_corr',
            'Figure 12. Adopted-input rank correlations: equal-date means of daily '
            'cross-sectional Spearman correlations in 2021–2025, sampled every fifth '
            'session in the common pool. The lower triangle uses color rather than '
            'a numerical label in every cell.'),
        _report_paragraph(
            f'T3 and T5 have rank correlation '
            f'{ctx["factor_corr"].loc["T3", "T5"]:+.2f}, showing overlap between volatility '
            'and activity. Correlated inputs need not be removed mechanically, but they '
            'can repeat the same exposure. Ridge regularization controls coefficients; '
            'it does not create independent information.'),
    ])
    pages.append([
        _report_paragraph('11. Forecast rank correlation', 'heading'), ('gap', 10),
        _report_paragraph(
            'Rank IC is the within-date Spearman correlation of a model score with '
            'the future 20-session opening return. A positive value means the full '
            'cross section is ordered in the intended direction on average; it does '
            'not directly describe the Top-100 trading portfolio.'), ('gap', 8),
        *_report_chart('rank_ic',
            'Figure 13. Annual mean daily Spearman rank IC of adopted scores against '
            'mature 20-session opening-return labels. The 2021–2025 values require '
            'label exits by 31 Dec 2025; 2026 exits end on 29 Apr 2026.'),
        _report_paragraph(
            f'Ridge’s 2026 mean IC is '
            f'{ctx["ic_annual"].loc[ctx["ic_annual"]["year"].eq(2026) & ctx["ic_annual"]["method"].eq("ridge"), "mean_ic"].iloc[0]:+.4f}, '
            'but a positive whole-pool IC can coexist with weak high-score selection. '
            f'The 2026 average uses {int(summary["labelled_2026_days"])} mature signal '
            f'dates through {label_last}. Overlapping labels make daily IC observations '
            'dependent; the chart is descriptive rather than an independent-day '
            'significance test. The next page examines the high-score groups directly.'),
    ])

    # Page 12: forward-label diagnostics are explicitly not tradable bucket backtests.
    quintile_spreads = {}
    for period in ['2021–2025', '2026']:
        grouped = ctx['quintiles'].loc[
            ctx['quintiles']['period'].eq(period) & ctx['quintiles']['method'].eq('ridge')
        ].set_index('group')
        quintile_spreads[period] = grouped.loc[5, 'mean_excess'] - grouped.loc[1, 'mean_excess']
    pages.append([
        _report_paragraph('12. Does the high-score group contain better returns?', 'heading'), ('gap', 10),
        *_report_chart('quintile',
            'Figure 14. Mean future 20-session return excess by score quintile, '
            '2021–2025. Q1 is the lowest score; Q5 is the highest.'),
        *_report_chart('quintile_2026',
            f'Figure 15. The same diagnostic for available 2026 signal dates through '
            f'{label_last}. Later signals do not yet have mature labels.'),
        _report_paragraph(
            'These are future-label cohort averages, not portfolio returns: they ignore '
            'fees, opening execution limits, cash and delayed sales. The comparison uses '
            'near-equal score groups before removing unavailable labels; ties use fixed '
            'security order. Excess subtracts the daily labeled score-pool mean, and '
            'daily cohort means receive equal weight. Labels ending after 2025 are '
            'excluded from the 2021–2025 panel. Adjacent cohorts overlap and are '
            'not independent. Ridge’s Q5-Q1 spread moves from '
            f'{_report_pp(quintile_spreads["2021–2025"])} pp to '
            f'{_report_pp(quintile_spreads["2026"])} pp, weakening high-score '
            'separation in 2026.'),
    ])

    # Frozen Lasso specification and development disclosure within the page budget.
    lasso_fits = ctx['lasso_fits'].loc[ctx['lasso_fits'].prediction_start.ge(20210101)].sort_values('year')
    lasso_rows = [['Year', 'Alpha', f'Nonzero / {lasso_settings["features"]}', 'Mean Rank IC']]
    for _, fit in lasso_fits.iterrows():
        year = int(fit['year'])
        ic = ctx['ic_annual'].loc[ctx['ic_annual'].year.eq(year) & ctx['ic_annual'].method.eq('lasso'), 'mean_ic'].iloc[0]
        lasso_rows.append([str(year), f'{fit["alpha"]:.3g}', str(int(fit['nonzero_count'])),
                           f'{ic:+.4f}' if np.isfinite(ic) else '— (no scores)'])
    comparison_rows = main_rows
    nonzero_min, nonzero_max = int(lasso_fits.nonzero_count.min()), int(lasso_fits.nonzero_count.max())
    boundary_count = int(lasso_fits.grid_boundary.sum())
    latest_coefficients = ctx['lasso_coefficients'].loc[ctx['lasso_coefficients'].year.eq(2026)]
    kept = latest_coefficients.loc[latest_coefficients.coefficient.ne(0), 'feature'].tolist()
    optimization_text = ''
    if optimization:
        optimization_text = (
            f'Outer search compared {optimization["development_candidates"]} configurations '
            f'({optimization["lasso_candidates"]} Lasso; {optimization["ridge_controls"]} Ridge controls). '
            f'Development data end {_report_date(str(optimization["selection_cutoff"]))}; '
            'the selected specification was frozen before 2026 reevaluation. '
            f'Original Lasso development net was {_report_percent(optimization["baseline_development_net"], 2)}. '
            f'Ridge-3y control (development only) returns {_report_percent(optimization["paired_ridge_development_net"], 2)} net. '
            'It matches Lasso inputs, window, target and EMA, using fixed alpha=10,000 '
            'and float64 Gram; it is not the main Ridge-exp curve. ')
        if optimization.get('baseline_holdout_net') is not None:
            optimization_text += (
                f'Prior Lasso 2026 net was {_report_percent(optimization["baseline_holdout_net"], 2)}, '
                f'versus {_report_percent(recent["lasso"].total_return, 2)} after optimization. ')
    pages.append([
        _report_paragraph('13. Lasso: sparse coefficients and matched returns', 'heading'), ('gap', 8),
        _report_paragraph(
            f'L1 regression uses {lasso_settings["features"]} inputs, {lasso_settings["window"]}, '
            f'a {lasso_settings["target"]} and score EMA{lasso_settings["span"]}. '
            'Top-100 timing, stock pool, execution and costs remain common. [5]'), ('gap', 8),
        _report_table(comparison_rows, [102, 106, 132, width - 340]), ('gap', 8),
        _report_paragraph('Historical selection and coefficient sparsity', 'subheading'),
        _report_paragraph(
            'Inner alpha validation uses the preceding calendar year: training-label exits '
            'precede validation, and validation-label exits precede the forecast. '
            f'Among {lasso_settings["alpha_count"]} training-only alpha-threshold fractions, '
            f'selection {lasso_settings["selector_action"]} {lasso_settings["selector"]} over rankable fits. '
            'The selected fraction scales the threshold recomputed on the final mature '
            'training window. Null and constant rankings remain diagnostics.'), ('gap', 8),
        _report_table(lasso_rows, [70, 110, 118, width - 298]), ('gap', 8),
        _report_paragraph(
            f'Annual fits retain {nonzero_min}–{nonzero_max} of {lasso_settings["features"]} inputs; '
            f'the 2026 nonzero inputs are {", ".join(kept)}. Correlated inputs can substitute '
            'for one another; prediction quality and fee-paying Top-100 returns differ.'), ('gap', 8),
        _report_paragraph(
            f'{optimization_text}{lasso_comparison} '
            f'{boundary_count} of six fits select an eligible-grid edge. '
            'Development gains do not establish L1 superiority. Already reviewed 2026 '
            'history is not a new blind test; it does not trigger reselection.'),
    ])

    # Page 14: limits and a conclusion that does not overstate evidence or style neutrality.
    final_blocks = [
        _report_paragraph('14. Limitations and conclusions', 'heading'), ('gap', 10),
        _report_paragraph('What the results support', 'subheading'),
        _report_paragraph(
            'Ridge and Lasso compare recorded specifications under matched execution. '
            + ('The auxiliary Ridge-3y control matches selected inputs, window, target '
               'and processing. ' if optimization else '')
            + 'Neither penalty guarantees stronger future returns or protection from drawdowns.'), ('gap', 8),
        _report_paragraph('What remains uncertain', 'subheading'),
        _report_paragraph(
            'Historical specification choices can favor patterns that do not persist. '
            'The reviewed 2026 record is short. Overlapping labels are dependent; '
            'complete-input filtering changes the stock pool; vendor revisions remain '
            'uncertain even with announcement-date guards.'), ('gap', 8),
        _report_paragraph(
            'Daily bars omit queue priority and precise market impact. Corporate-action '
            'prices approximate reinvestment rather than exact dividend receipts and taxes. '
            'Residualizing scores does not enforce neutral actual holdings or lower risk.'), ('gap', 8),
    ]
    if 'cap_weights' in charts:
        final_blocks.extend(_report_chart('cap_weights',
            f'Figure 17. Actual net-NAV weights on {end_date} by market-cap quintile '
            '(Q1 small, Q5 large), using same-day eligible-pool boundaries. Other '
            'combines cash, pending claims and unknown capitalization.'))
    final_blocks.append(_report_paragraph(
        'The historical net result warrants further study. Ridge’s '
        f'{_report_percent(recent["ridge"]["total_return"], 2)} net return in 2026 through '
        f'{end_date}, against {_report_percent(recent["benchmark"]["total_return"], 2)} '
        'for the matched benchmark, is modest evidence over a partial year. '
        'A future prospective run with unchanged rules would provide a stronger check.'))
    pages.append(final_blocks)

    # References are separate from the fifteen-page main-body budget.
    pages.append([
        _report_paragraph('References and reproducibility', 'heading'), ('gap', 10),
        _report_paragraph(
            '[1] Scikit-learn. <i>Ridge regression documentation.</i> Objective and '
            'L2-regularization parameter. '
            '<link href="https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.Ridge.html" '
            'color="#137F82">Official API documentation</link>.'), ('gap', 10),
        _report_paragraph(
            '[2] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., '
            'and Liu, T.-Y. (2017). <i>LightGBM: A Highly Efficient Gradient Boosting '
            'Decision Tree.</i> Advances in Neural Information Processing Systems 30. '
            '<link href="https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html" '
            'color="#137F82">Conference paper</link>.'), ('gap', 10),
        _report_paragraph(
            '[3] Ministry of Finance, China (2008). <i>Securities transaction stamp duty '
            'changed to sell-side collection from 19 September 2008.</i> '
            '<link href="https://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/200809/t20080919_76432.htm" '
            'color="#137F82">Official announcement</link>.'), ('gap', 10),
        _report_paragraph(
            '[4] Ministry of Finance and State Taxation Administration, China (2023). '
            '<i>Announcement No. 39: Halving securities transaction stamp duty from '
            '28 August 2023.</i> '
            '<link href="https://m.mof.gov.cn/czxw/202308/t20230827_3904226.htm" '
            'color="#137F82">Official announcement</link>. Commission and slippage '
            'are research assumptions.'), ('gap', 12),
        _report_paragraph(
            '[5] Scikit-learn. <i>Lasso regression documentation.</i> L1 objective, '
            'coefficient sparsity and convergence diagnostics. '
            '<link href="https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.Lasso.html" '
            'color="#137F82">Official API documentation</link>.'), ('gap', 10),
        _report_paragraph('Reproduction from prepared external inputs', 'subheading'),
        _report_paragraph(
            'config.json records fixed annual factor selections, directions, model settings '
            'and trading assumptions. data_dictionary.md defines the prepared arrays, factor '
            'formulas, financial visibility guards and external corporate-action table. '
            'The calculation code retrains the fixed models and reconstructs all selected '
            'and raw-baseline portfolios from those inputs; no previous score or return '
            'file is required as a model input.'), ('gap', 8),
        _report_paragraph(
            'Generated external results contain the continuous daily ledger and metrics for '
            'the 20-session main implementation, daily comparison and daily doubled-slippage '
            'comparison. The raw baseline has its own matched 20-session ledger. Charts '
            'use those ledgers, while rank-IC and score-group figures use mature labels '
            'and adopted scores. Obtaining or rebuilding the supplier’s original financial '
            'snapshots is outside the prepared-input reproduction.'), ('gap', 8),
    ])

    temp = output_root / 'report_assets'
    temp.mkdir(parents=True, exist_ok=True)
    base_path = temp / 'report_text.pdf'
    pdf_path = project / 'Report_Group9.pdf'
    canvas = _report_Canvas(str(base_path), pagesize=_report_A4)
    canvas.setTitle('Fundamental and technical signals for A-share stock selection | Group 9')
    canvas.setAuthor('Investments 1005 | Group 9')
    chart_boxes, bounds = [], []
    for page_index, blocks in enumerate(pages):
        canvas.setFillColor(_report_colors.HexColor(_report_muted))
        canvas.setFont('Times-Roman', 9)
        canvas.drawString(72, page_h - 42, 'Investments 1005 | Group 9')
        canvas.drawRightString(page_w - 72, 42, str(page_index + 1))
        y = page_h - 72
        for block in blocks:
            previous = y
            if block[0] == 'gap':
                y -= block[1]
            elif block[0] == 'p':
                paragraph = _report_Paragraph(block[2], styles[block[1]])
                _, height = paragraph.wrap(width, page_h)
                y -= height
                paragraph.drawOn(canvas, 72, y)
            elif block[0] == 'table':
                table = _report_render_table(block[1], block[2], styles)
                _, height = table.wrap(width, page_h)
                y -= height
                table.drawOn(canvas, 72, y)
            else:
                chart = charts[block[1]]['report']
                native_width = float(chart['width']) * 72
                native_height = float(chart['height']) * 72
                # Preserve the exported aspect ratio and its 12-point labels.
                assert abs(native_width - width) < .05, (block[1], native_width, width)
                rect = _report_fitz.Rect(72, page_h - y, page_w - 72, page_h - y + native_height)
                chart_boxes.append((page_index, Path(chart['pdf']), rect))
                y -= native_height
            bounds.append({
                'page': page_index + 1, 'kind': block[0],
                'top': page_h - previous, 'bottom': page_h - y,
            })
            assert y >= 72, (page_index + 1, block[0], 'Report content exceeds one-inch bottom margin', y)
        canvas.showPage()
    canvas.save()
    with _report_fitz.open(base_path) as document:
        for page_index, chart_path, rect in chart_boxes:
            with _report_fitz.open(chart_path) as chart_document:
                document[page_index].show_pdf_page(rect, chart_document, 0)
        document.save(pdf_path, garbage=4, deflate=True)
    return {
        'path': str(pdf_path), 'body_pages': len(pages) - 1, 'total_pages': len(pages),
        'body_font_pt': 12, 'body_leading_pt': 18, 'margins_pt': 72,
        'blocks': bounds,
    }


"""Editable Group 9 research presentation; native text and shared PDF geometry."""

from io import BytesIO as _ppt_BytesIO
from pathlib import Path as _ppt_Path
from datetime import date as _ppt_date_type
import fitz as _ppt_fitz
import matplotlib as _ppt_matplotlib
from pptx import Presentation as _ppt_Presentation
from pptx.util import Inches as _ppt_Inches, Pt as _ppt_Pt
from pptx.dml.color import RGBColor as _ppt_RGBColor
from pptx.enum.shapes import MSO_SHAPE as _ppt_MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR as _ppt_MSO_ANCHOR, PP_ALIGN as _ppt_PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement as _ppt_OxmlElement
from pptx.oxml.ns import qn as _ppt_qn
from reportlab.pdfgen.canvas import Canvas as _ppt_Canvas
from reportlab.pdfbase import pdfmetrics as _ppt_pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont as _ppt_TTFont
from reportlab.lib import colors as _ppt_colors

_ppt_ink = "#17212D"
_ppt_teal = "#007C83"
_ppt_muted = "#68737E"
_ppt_grid = "#D9DEE3"
_ppt_width = 13.333333
_ppt_height = 7.5
_ppt_methods = ("fundamental", "combined", "ridge", "lasso", "lightgbm", "benchmark")
_ppt_labels = {"fundamental": "Fundamental", "combined": "Combined", "ridge": _MAIN_MODEL_VERSIONS["ridge"], "lasso": _MAIN_MODEL_VERSIONS["lasso"], "lightgbm": "LightGBM", "benchmark": "Benchmark"}
_ppt_seconds = [20,70,50,55,75,65,50,60,40,65,75,55,45,40,50,60,55,55,65,30,0,0,0]

def _ppt_register_fonts():
    folder = _ppt_Path(_ppt_matplotlib.get_data_path()) / "fonts/ttf"
    _ppt_pdfmetrics.registerFont(_ppt_TTFont("G9Slide", str(folder / "DejaVuSans.ttf")))
    _ppt_pdfmetrics.registerFont(_ppt_TTFont("G9SlideBold", str(folder / "DejaVuSans-Bold.ttf")))


def _ppt_wrap(text, width, size, bold=False):
    font = "G9SlideBold" if bold else "G9Slide"
    lines = []
    for paragraph in str(text).split("\n"):
        line = ""
        for word in paragraph.split():
            trial = (line + " " + word).strip()
            if line and _ppt_pdfmetrics.stringWidth(trial, font, size) > width * 72:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    return lines


def _ppt_text(text, x, y, w, h, size=20, bold=False, color=_ppt_ink, align="left"):
    return {"type": "text", "text": str(text), "x": x, "y": y, "w": w, "h": h,
            "size": size, "bold": bold, "color": color, "align": align}


def _ppt_metric(ctx, method, stage, field="total_return", fee="net"):
    frame = ctx["metrics"]
    row = frame[(frame["scope"] == "rebalance20") & (frame["stage"] == stage)
                & (frame["method"] == method) & (frame["fee"] == fee)].iloc[0]
    return float(row[field])


def _ppt_percent(value, digits=2, signed=False):
    return format(value, ("+" if signed else "") + f".{digits}%")


def _ppt_pp(value, digits=2):
    return f"{value * 100:+.{digits}f} pp"


def _ppt_human_date(value):
    return _ppt_date_type.fromisoformat(str(value)[:10]).strftime("%d %b %Y").lstrip("0")


def _ppt_draw_text(slide, canvas, item):
    lines = _ppt_wrap(item["text"], item["w"], item["size"], item["bold"])
    leading = item["size"] * 1.12
    needed = item["size"] + (len(lines) - 1) * leading
    if needed > item["h"] * 72 + .2:
        raise ValueError(f"Slide text exceeds its box ({needed:.1f}pt): {item['text']}")
    font = "G9SlideBold" if item["bold"] else "G9Slide"
    for line in lines:
        if _ppt_pdfmetrics.stringWidth(line, font, item["size"]) > item["w"] * 72 + .2:
            raise ValueError(f"Unbreakable slide text exceeds its box: {line}")
    shape = slide.shapes.add_textbox(_ppt_Inches(item["x"]), _ppt_Inches(item["y"]),
                                    _ppt_Inches(item["w"]), _ppt_Inches(item["h"]))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = False
    frame.margin_top = frame.margin_bottom = frame.margin_left = frame.margin_right = 0
    frame.vertical_anchor = _ppt_MSO_ANCHOR.TOP
    para = frame.paragraphs[0]
    para.text = "\v".join(lines)
    para.font.name = "DejaVu Sans"
    para.font.size = _ppt_Pt(item["size"])
    para.font.bold = item["bold"]
    para.font.color.rgb = _ppt_RGBColor.from_string(item["color"][1:])
    para.space_before = para.space_after = _ppt_Pt(0)
    para.line_spacing = _ppt_Pt(leading)
    para.alignment = {"left": _ppt_PP_ALIGN.LEFT, "center": _ppt_PP_ALIGN.CENTER,
                      "right": _ppt_PP_ALIGN.RIGHT}[item["align"]]
    canvas.setFillColor(_ppt_colors.HexColor(item["color"]))
    canvas.setFont(font, item["size"])
    baseline = _ppt_height * 72 - item["y"] * 72 - item["size"] * .86
    for line in lines:
        x = item["x"] * 72
        if item["align"] == "center":
            canvas.drawCentredString(x + item["w"] * 36, baseline, line)
        elif item["align"] == "right":
            canvas.drawRightString(x + item["w"] * 72, baseline, line)
        else:
            canvas.drawString(x, baseline, line)
        baseline -= leading
    return len(lines)


def _ppt_header(title, subtitle=""):
    items = [_ppt_text(title, .85, .52, 11.65, .60, 30, True)]
    if subtitle:
        items.append(_ppt_text(subtitle, .85, 1.21, 11.65, .47, 16, color=_ppt_muted))
    return items


def _ppt_chart(charts, key, frame=(.85, 1.68, 11.65, 4.60)):
    """Center the actual exported physical size; never resize a chart to fit."""
    art = charts[key]["slide"]
    width, height = float(art["width"]), float(art["height"])
    x, y, max_width, max_height = frame
    if width > max_width + .01 or height > max_height + .01:
        raise ValueError(f"{key}: export {width} x {height} exceeds its {max_width} x {max_height} inch frame")
    if key in ("gross_absolute", "net_absolute", "gross_excess", "net_excess") and width / height > 3.2:
        raise ValueError(f"{key}: the main return chart is too flat; regenerate the figure at about 10.6 x 4.2 inches")
    with _ppt_fitz.open(art["pdf"]) as source:
        rect = source[0].rect
        if abs(rect.width / 72 - width) > .02 or abs(rect.height / 72 - height) > .02:
            raise ValueError(f"{key}: chart PDF physical size differs from its manifest")
    return {"type": "chart", "key": key, "png": str(art["png"]), "pdf": str(art["pdf"]),
            "x": x + (max_width - width) / 2, "y": y + (max_height - height) / 2,
            "w": width, "h": height}


def _ppt_table(rows, x, y, widths, row_height=.53, size=18, numeric_columns=(), highlight_rows=()):
    return {"type": "table", "rows": rows, "x": x, "y": y, "w": sum(widths),
            "h": len(rows) * row_height, "widths": widths, "row_height": row_height,
            "size": size, "numeric_columns": list(numeric_columns), "highlight_rows": list(highlight_rows)}


def _ppt_cell_border(cell):
    properties = cell._tc.get_or_add_tcPr()
    for edge in ("lnL", "lnR", "lnT", "lnB"):
        existing = properties.find(_ppt_qn("a:" + edge))
        if existing is not None:
            properties.remove(existing)
        line = _ppt_OxmlElement("a:" + edge)
        line.set("w", "6350")
        if edge == "lnB":
            fill = _ppt_OxmlElement("a:solidFill")
            colour = _ppt_OxmlElement("a:srgbClr")
            colour.set("val", _ppt_grid[1:])
            fill.append(colour)
            line.append(fill)
        else:
            line.append(_ppt_OxmlElement("a:noFill"))
        properties.append(line)


def _ppt_draw_table(slide, canvas, item):
    rows, widths = item["rows"], item["widths"]
    height, size = item["row_height"], item["size"]
    shape = slide.shapes.add_table(len(rows), len(widths), _ppt_Inches(item["x"]),
                                  _ppt_Inches(item["y"]), _ppt_Inches(item["w"]), _ppt_Inches(item["h"]))
    table = shape.table
    table.first_row = table.horz_banding = table.vert_banding = False
    for column, width in zip(table.columns, widths):
        column.width = _ppt_Inches(width)
    for row in table.rows:
        row.height = _ppt_Inches(height)
    for r, values in enumerate(rows):
        x = item["x"]
        bold = r == 0 or r in item["highlight_rows"]
        colour = _ppt_teal if r in item["highlight_rows"] else _ppt_ink
        font = "G9SlideBold" if bold else "G9Slide"
        for c, (value, width) in enumerate(zip(values, widths)):
            fill = "#F2F4F5" if r == 0 else "#FFFFFF"
            cell = table.cell(r, c)
            _ppt_cell_border(cell)
            cell.fill.solid()
            cell.fill.fore_color.rgb = _ppt_RGBColor.from_string(fill[1:])
            cell.margin_left = cell.margin_right = _ppt_Inches(.09)
            cell.margin_top = _ppt_Inches(.10)
            cell.margin_bottom = 0
            cell.vertical_anchor = _ppt_MSO_ANCHOR.TOP
            frame = cell.text_frame
            frame.clear()
            frame.word_wrap = False
            lines = _ppt_wrap(value, width - .18, size, bold)
            if size + (len(lines) - 1) * size * 1.1 > (height - .12) * 72:
                raise ValueError(f"Table cell overflows: {value}")
            p = frame.paragraphs[0]
            p.text = "\v".join(lines)
            p.font.name = "DejaVu Sans"
            p.font.size = _ppt_Pt(size)
            p.font.bold = bold
            p.font.color.rgb = _ppt_RGBColor.from_string(colour[1:])
            p.space_before = p.space_after = _ppt_Pt(0)
            p.line_spacing = _ppt_Pt(size * 1.1)
            right = c in item["numeric_columns"]
            p.alignment = _ppt_PP_ALIGN.RIGHT if right else _ppt_PP_ALIGN.LEFT
            canvas.setFillColor(_ppt_colors.HexColor(fill))
            canvas.rect(x * 72, (_ppt_height - item["y"] - (r + 1) * height) * 72,
                        width * 72, height * 72, fill=1, stroke=0)
            canvas.setFillColor(_ppt_colors.HexColor(colour))
            canvas.setFont(font, size)
            for number, line in enumerate(lines):
                baseline = (_ppt_height - item["y"] - r * height - .10) * 72 - size * .86 - number * size * 1.1
                if right:
                    canvas.drawRightString((x + width - .09) * 72, baseline, line)
                else:
                    canvas.drawString((x + .09) * 72, baseline, line)
            canvas.setStrokeColor(_ppt_colors.HexColor(_ppt_grid))
            canvas.setLineWidth(.5)
            canvas.line(x * 72, (_ppt_height - item["y"] - (r + 1) * height) * 72,
                        (x + width) * 72, (_ppt_height - item["y"] - (r + 1) * height) * 72)
            x += width


def _ppt_plot_page(charts, key, title, subtitle, takeaway):
    return [*_ppt_header(title, subtitle), _ppt_chart(charts, key),
            _ppt_text(takeaway, .98, 6.43, 11.35, .62, 20)]


def write_presentation(ctx, charts, project, output_root):
    """Write a white research deck: 20 main slides, 3 untimed appendices.

    Charts retain their exported physical sizes. Text, numerical summaries and
    tables are editable PPT objects. The PDF uses the same element geometry
    with the corresponding chart PDF pages, independently of Office.
    """
    _ppt_register_fonts()
    project = _ppt_Path(project)
    project.mkdir(parents=True, exist_ok=True)
    summary = ctx["summary"]
    lasso_settings = _lasso_report_settings(ctx["config"])
    optimization = ctx["config"].get("lasso_optimization")
    latest = int(summary["latest_year"])
    end = _ppt_human_date(summary["end"])
    span = f"{_ppt_human_date(summary['start'])} – {end}"
    ridge_dev = _ppt_metric(ctx, "ridge", "development")
    pool_dev = _ppt_metric(ctx, "benchmark", "development")
    ridge_latest = _ppt_metric(ctx, "ridge", "holdout")
    latest_excess = _ppt_metric(ctx, "ridge", "holdout", "cumulative_excess")
    dd = _ppt_metric(ctx, "ridge", "full", "max_drawdown")
    pool_dd = _ppt_metric(ctx, "benchmark", "full", "max_drawdown")
    lasso_differences = [_ppt_metric(ctx, 'lasso', stage) - _ppt_metric(ctx, 'ridge', stage)
                         for stage in ('development', 'holdout')]
    lasso_comparison = ('Lasso exceeds original Ridge in both intervals.' if min(lasso_differences) > 0
                        else 'Lasso trails original Ridge in both intervals.' if max(lasso_differences) < 0
                        else 'Lasso improves development return; 2026 trails original Ridge.'
                        if lasso_differences[0] > 0 and lasso_differences[1] < 0
                        else 'Lasso–original Ridge results vary across intervals.')
    pages, notes = [], []

    # 1. A cover, with no performance tiles or dense table.
    pages.append([
        _ppt_text("INVESTMENTS 1005  /  GROUP 9", .85, .72, 11.5, .3, 14, True, _ppt_muted),
        {"type": "rule", "x": .85, "y": 1.27, "w": 1.15, "h": .035, "color": _ppt_teal},
        _ppt_text("A-share cross-sectional\nstock selection", .85, 1.85, 11.6, 1.45, 36, True),
        _ppt_text("Fundamentals, market signals and machine learning", .88, 3.74, 11.5, .55, 22),
        _ppt_text(f"{span}\nDaily inputs · Top 100 · 20-session targets", .88, 5.22, 11.5, .83, 20, color=_ppt_muted),
    ])
    notes.append("Introduce the question: can financial and market signals improve cross-sectional stock selection after stated trading costs? Identify five methods, the common opportunity set and the observed historical dates.")

    # The same six ledger records drive the headline table in both documents.
    comparison = _main_comparison(ctx)
    main_rows = [["Model version", "Full net", "2021–2025 net", f"{latest} net"]]
    for version in _MAIN_MODEL_VERSIONS.values():
        values = comparison.loc[comparison.model_version.eq(version)].set_index('period')
        main_rows.append([version, *[_ppt_percent(values.loc[period, 'net_return'])
                                    for period in _MAIN_PERIODS]])
    items = _ppt_header("One comparison, three periods",
                        "The same published model versions in the tables, curves and conclusions")
    items += [_ppt_table(main_rows, .85, 2.06, [3.00, 2.80, 2.90, 2.90], .66, 23, [1, 2, 3]),
              _ppt_text("Ridge-exp: original expanding history · Lasso-3y: optimized three-year window", .98, 4.55, 11.35, .50, 19),
              _ppt_text("Full history includes 2018–2020 fitted replay; 2026 ends on 29 April.", .98, 5.27, 11.35, .45, 19),
              _ppt_text("Ridge-3y control is a separate development diagnostic, absent from the main curves.", .98, 5.96, 11.35, .45, 18, color=_ppt_muted),
              _ppt_text("Lasso-3y leads full/development; Ridge-exp leads 2026.", .98, 6.64, 11.35, .40, 20, True)]
    pages.append(items)
    notes.append(
        "Read the same two published models across all three periods: full Lasso-3y "
        f"{_ppt_percent(_ppt_metric(ctx, 'lasso', 'full'))}, Ridge-exp "
        f"{_ppt_percent(_ppt_metric(ctx, 'ridge', 'full'))}; development "
        f"{_ppt_percent(_ppt_metric(ctx, 'lasso', 'development'))} versus "
        f"{_ppt_percent(_ppt_metric(ctx, 'ridge', 'development'))}; 2026 "
        f"{_ppt_percent(_ppt_metric(ctx, 'lasso', 'holdout'))} versus "
        f"{_ppt_percent(_ppt_metric(ctx, 'ridge', 'holdout'))}. "
        "Full history includes fitted replay. These are not different model versions. "
        "The auxiliary Ridge-3y control is a development diagnostic and is not the main Ridge-exp curve.")

    # 3. Two text columns instead of three decorative cards.
    pages.append([
        *_ppt_header("Why combine these signals?", f"Fourteen candidate inputs; {len(summary['selected_features'])} used across the frozen models"),
        _ppt_text("Fundamentals", 1.00, 1.96, 5.25, .4, 23, True),
        _ppt_text("Profitability and cash generation\n\nSigned book and earnings yields\n\nGrowth and leverage dynamics", 1.00, 2.68, 5.25, 2.45, 21),
        _ppt_text("F1 · F2 · F3 · F5 · F6 · F7 · F8 · F9 · F10", 1.00, 5.54, 5.25, .55, 14, color=_ppt_muted),
        _ppt_text("Market behaviour", 7.05, 1.96, 5.25, .4, 23, True),
        _ppt_text("20- and 60-session momentum\n\n126-to-20-session momentum\n\nReturn volatility and turnover", 7.05, 2.68, 5.25, 2.45, 21),
        _ppt_text("T1 · T2 · T3 · T5 · T6", 7.05, 5.54, 5.25, .4, 14, color=_ppt_muted),
        _ppt_text("Distinct mechanisms motivate the inputs; their usefulness is empirical.", .98, 6.48, 11.35, .5, 20),
    ])
    notes.append("Explain profitability, cash conversion, signed value and balance-sheet dynamics, then momentum, volatility and turnover. Negative book values and earnings are retained. F4 and T4 remain saved columns but are unused by the pool and final models. Candidate mechanisms motivate the study; fixed annual selections and directions come from the published configuration, not a new search.")

    pages.append(_ppt_plot_page(charts, "coverage", "Data and the common stock pool",
        f"{int(summary['days']):,} sessions · {int(summary['stock_slots']):,} historical slots · First signal: {_ppt_human_date(summary['first_signal'])}",
        "The same fourteen-input finite pool is used by every portfolio."))
    notes.append("Describe historical membership, non-ST and listing-age screens, ordinary A-share codes, trading status and the finite fourteen-input common mask. The original sixteen-column panel is preserved. Report period and both announcement dates are checked against the signal day, while later supplier revisions cannot be independently excluded. Missing inputs stay missing, and future label availability never determines current eligibility.")

    # 5. A plain sequence and a legible native model table.
    pipeline = []
    for x, text in zip([.95, 3.40, 5.85, 8.30, 10.75],
                       ["Input\nranks", "Annual\nmodel", "Style\nresiduals", "Causal\nEMA", "Score\nranks"]):
        pipeline.append(_ppt_text(text, x, 1.98, 1.65, .72, 20, True, _ppt_ink, "center"))
        if x < 10:
            pipeline.append(_ppt_text("→", x + 1.91, 2.12, .3, .4, 20, color=_ppt_muted))
    model_rows = [["Method", "Score or model", "EMA sessions"],
                  ["Fundamental", "Signed financial ranks", "20"],
                  ["Combined", "Equal financial / technical families", "40"],
                  ["Ridge-exp", "Expanding history; alpha = 10,000", "20"],
                  ["Lasso-3y", f"{lasso_settings['features']} inputs; 3-year history; validation alpha", str(lasso_settings['span'])],
                  ["LightGBM", "depth 3; 7 leaves; 56 trees", "40"]]
    pages.append([*_ppt_header("Fixed models, annual refits", "Predict 20-session returns; only labels with earlier t+21 exits enter training"),
                  *pipeline, _ppt_table(model_rows, .85, 3.12, [2.7, 6.0, 2.9], .50, 18, [2], [3]),
                  _ppt_text("Industry and log-size residualization precede the causal score EMA.", .98, 6.48, 11.35, .5, 20)])
    notes.append("Walk through input ranks, the annual fit, industry and within-industry log-capitalization residualization, and the causal EMA followed by ranks. State the Ridge alpha and shallow-tree settings. Training uses the absolute every-fifth-session grid and a strictly mature t+21 exit. The 2018–2020 history is fitted replay. From 2021, annual predictions use mature prior labels. Returns and NAV are not smoothed.")

    cost_rows = [["Commission", "Slippage", "Sell stamp tax"],
                 ["3 bp per side", "5 bp per side", "10 → 5 bp on 28 Aug 2023"]]
    pages.append([
        *_ppt_header("Portfolio construction and execution", "Top 100 and equal-weight pool portfolios use the same implementation rules"),
        _ppt_text("Selection and timing", 1.00, 1.96, 5.4, .4, 23, True),
        _ppt_text("Equal-weight Top 100\nContinuous 20-session targets\nClose signals, next-open trading\nT+1, limits and order retries", 1.00, 2.67, 5.4, 1.80, 21),
        _ppt_text("Matched benchmark", 7.10, 1.96, 5.2, .4, 23, True),
        _ppt_text("Equal-weight common stock pool\nSame cadence and cash constraints\nSame restrictions and costs\nCSI500 is an auxiliary reference", 7.10, 2.67, 5.2, 1.80, 21),
        _ppt_table(cost_rows, .85, 5.08, [3.5, 3.5, 4.6], .58, 18),
        _ppt_text("Twelve sourced company events use explicit claims and settlement dates.", .98, 6.55, 11.35, .4, 18, color=_ppt_muted),
    ])
    notes.append("Differentiate daily signals and valuation from twenty-session targets. Gross and net books hold separate cash and securities. Explain T+1, opening limits, suspended orders and retries. The benchmark uses the same pool, frequency, rules and matching fees. Explain exchange and cash claims, cash released only after payment, documented settlement-date assumptions, and the zero-recovery end-date policy for unresolved positions. Synthetic adjusted units approximate dividend reinvestment.")

    pages.append(_ppt_plot_page(charts, "gross_absolute", "Gross absolute return", span,
        f"Full gross: Lasso-3y {_ppt_percent(_ppt_metric(ctx, 'lasso', 'full', fee='gross'))}; Ridge-exp {_ppt_percent(_ppt_metric(ctx, 'ridge', 'full', fee='gross'))}."))
    notes.append("Read the complete gross return history. Include all five methods and the pool rather than only the winner. Wealth compounds daily portfolio returns. The initial flat section precedes feasible trading, and the early history is fitted replay. There is no yearly restart or clipped weak interval.")

    pages.append(_ppt_plot_page(charts, "net_absolute", "Net absolute return", span,
        f"Full net: Lasso-3y {_ppt_percent(_ppt_metric(ctx, 'lasso', 'full'))}; Ridge-exp {_ppt_percent(_ppt_metric(ctx, 'ridge', 'full'))}."))
    notes.append("Compare this net chart with the preceding gross chart. Costs are charged inside the cash ledger and influence later holdings, so the compounded gross-minus-net wealth difference is not a pure fee sum. Keep every method visible and use the table for the weaker latest-year results.")

    pages.append(_ppt_plot_page(charts, "gross_excess", "Gross excess return", f"{span} · Daily differences versus the gross pool (pp)",
        "The pool's own excess is zero; absolute wealth uses a different calculation."))
    notes.append("Excess is an arithmetic sum of daily strategy-minus-matched-pool returns, expressed in percentage points. The pool has zero excess against itself and is explicitly identified. This is not compounded relative wealth or the difference between compounded terminal returns.")

    pages.append(_ppt_plot_page(charts, "net_excess", "Net excess return", f"{span} · Daily differences versus the net pool (pp)",
        f"Full net excess: Lasso-3y {_ppt_pp(_ppt_metric(ctx, 'lasso', 'full', 'cumulative_excess'))}; Ridge-exp {_ppt_pp(_ppt_metric(ctx, 'ridge', 'full', 'cumulative_excess'))}."))
    notes.append("Read both strong and weak portions of net excess. The matched pool pays the same cost schedule and obeys the same execution rules. In 2026, Fundamental and LightGBM lag while Combined and Ridge have positive summed excess. The latest period is incomplete; do not present the full-history positive endpoint as evidence that every interval wins.")

    pages.append([
        *_ppt_header("Return comes with drawdown risk", "Same published versions · Full-history drawdown; 2021–2025 return / volatility"),
        _ppt_text("DRAWDOWN", .98, 1.78, 5.25, .3, 17, True),
        _ppt_text("RETURN / RISK", 7.12, 1.78, 5.25, .3, 17, True),
        _ppt_chart(charts, "drawdown", (.85, 2.08, 5.5, 4.15)),
        _ppt_chart(charts, "risk_return", (6.98, 2.08, 5.5, 4.15)),
        _ppt_text("Risk scatter: Ridge-exp versus Lasso-3y; auxiliary Ridge-3y control is not plotted.", .98, 6.55, 11.35, .45, 20),
    ])
    notes.append("Compare peak-to-trough net wealth losses with the development return-risk scatter. The panels state their different date ranges. Ridge's full-history maximum decline is slightly deeper than the pool, so higher return is not lower risk. Explain annualized volatility and a zero risk-free-rate Sharpe convention if asked.")

    pages.append(_ppt_plot_page(charts, "calendar_excess", "Monthly net excess", "Ridge-exp · Sum of daily net differences within each month, in pp",
        f"{latest} ends on 29 April; later months remain unobserved."))
    notes.append("Read positive and negative monthly cells. Months sum daily net return differences, and future months are blank or dashed rather than zero. The latest year stops on 29 April. Discuss visible weak months as well as strong runs; the calendar is descriptive, not an independent repeated experiment.")

    pages.append(_ppt_plot_page(charts, "annual_return", "Calendar-year net return", "Ridge-exp and the matched pool · Compounded within each calendar year",
        f"The {latest} bar covers observations only through {end}."))
    notes.append("Annual absolute returns compound within the calendar year, unlike excess sums. Keep negative years visible. Compare 2026 as a partial-year observation rather than a full twelve-month result. Annual portfolio values come from the continuous ledger rather than a new January portfolio reset.")

    pages.append(_ppt_plot_page(charts, "turnover", "Monthly portfolio turnover", "Half of traded notional divided by opening wealth; summed within month",
        "Twenty-session targets retain blocked-order retries and company-event trades."))
    notes.append("Define the one-half turnover convention and monthly aggregation. Explain why turnover is not zero between scheduled target changes: fills can retry and company events can settle. Gross and net holdings may diverge as costs change available cash. Do not equate the signal EMA span with the rebalance interval.")

    pages.append(_ppt_plot_page(charts, "costs", "Trading costs by calendar year", "Annual sums of daily cost / opening NAV (pp); 2026 through 29 April",
        "Cost-to-NAV sums and compounded terminal wealth measure different quantities."))
    notes.append("Read annual stacked commission, slippage and stamp-tax ratios. Each component sums cost divided by that day's opening net NAV, in percentage points. The 2026 column is partial year. These annual sums are not the compounded difference between terminal gross and net NAV. Costs affect cash and subsequent positions, and stamp tax changes on 28 August 2023.")

    pages.append(_ppt_plot_page(charts, "rank_ic", "Cross-sectional prediction quality", "Annual mean Spearman IC · Mature labels; 2026 signals through 30 March",
        "Full-rank prediction quality can coexist with weak top-ranked returns."))
    notes.append("IC measures the entire eligible cross-sectional ranking, not the return of the top hundred holdings. Average ties are retained. Use only labels with observable mature exits, and restrict development diagnostic exits to 2025 year-end. The 2026 signal-label window is shorter. Positive IC need not imply a profitable fee-paying portfolio or a strong extreme tail.")

    pages.append(_ppt_plot_page(charts, "quintile", "Score quintiles: 2021–2025", "Mean 20-session same-pool excess before costs, in pp; overlapping cohorts",
        "Higher development scores associate with higher cohort excess."))
    notes.append("Quintile 1 has low scores and 5 high scores. Groups are assigned before label availability is inspected, with a fixed security-slot tie break. Each original group then averages finite mature labels, minus the observable same-day pool mean. Development exits do not go beyond 2025. Means weight signal dates equally. These overlapping gross cohorts are not a tradable portfolio NAV.")

    pages.append(_ppt_plot_page(charts, "quintile_2026", f"Score quintiles: {latest}",
        f"{int(summary['labelled_2026_days'])} mature signal dates; last signal {_ppt_human_date(summary['last_label_signal'])}; same-pool excess, pp",
        "The latest-year relationship is weak and non-monotone."))
    notes.append("Show the weaker latest-year diagnostic without hiding it. Only signals whose twenty-session exit is observed by 29 April enter the panel; the final mature signal is 30 March. Explain the limited fifty-five-date sample, overlapping labels and nearly complete observed-label coverage. In particular, the highest group need not beat the middle groups or the same-pool mean. This is signal diagnosis before trading costs.")

    lasso_fits = ctx['lasso_fits'].loc[ctx['lasso_fits'].prediction_start.ge(20210101)].sort_values('year')
    sparse_rows = [["Fit year", "Alpha", f"Nonzero / {lasso_settings['features']}", "Validation"]]
    for _, fit in lasso_fits.iterrows():
        sparse_rows.append([str(int(fit['year'])), f'{fit["alpha"]:.3g}',
                            str(int(fit['nonzero_count'])), str(int(fit['year']) - 1)])
    lasso_dev = _ppt_metric(ctx, 'lasso', 'development')
    lasso_latest = _ppt_metric(ctx, 'lasso', 'holdout')
    optimization_takeaway = 'Null fits remain diagnostics; rankable models follow the original Top-100 rule.'
    if optimization:
        optimization_takeaway = (
            f'{optimization["development_candidates"]} dev tests · Ridge-3y control '
            f'{_ppt_percent(optimization["paired_ridge_development_net"])} · '
            f'original Lasso {_ppt_percent(optimization["baseline_development_net"])}.')
        if optimization.get('baseline_holdout_net') is not None:
            optimization_takeaway += (
                f'\n2026: selected {_ppt_percent(lasso_latest)} vs original Lasso '
                f'{_ppt_percent(optimization["baseline_holdout_net"])}; no reselection.')
    pages.append([
        *_ppt_header("Lasso-3y: sparse model, matched experiment",
                     f'{lasso_settings["features"]} inputs · {lasso_settings["window"]} · '
                     f'{lasso_settings["target"]} · EMA{lasso_settings["span"]}'),
        _ppt_table(sparse_rows, .85, 1.85, [2.20, 2.55, 2.45, 4.40], .44, 18, [1, 2, 3]),
        _ppt_text(f"Net return: 2021–2025 { _ppt_percent(lasso_dev)}; {latest} {_ppt_percent(lasso_latest)}.", .98, 5.28, 11.35, .55, 22, True),
        _ppt_text(optimization_takeaway, .98, 6.13, 11.35, .65, 19),
    ])
    notes.append(
        f'Explain the frozen Lasso: {lasso_settings["features"]} inputs, '
        f'{lasso_settings["window"]}, {lasso_settings["target"]}, EMA{lasso_settings["span"]}. '
        f'Inner alpha selection {lasso_settings["selector_action"]} {lasso_settings["selector"]} '
        f'over {lasso_settings["alpha_count"]} fractions. Internal training exits precede '
        'prior-calendar-year validation; validation exits precede each forecast year. '
        'Only rankable fits can enter the same Top-100 strategy. '
        + (f'Outer selection compares {optimization["lasso_candidates"]} Lasso configurations '
           f'and {optimization["ridge_controls"]} paired Ridge controls on 2021–2025 net '
           'results, with paired inputs, window, target and EMA. Ridge-3y control uses fixed '
           'alpha10,000 and a float64 Gram solver; its penalty is not separately tuned. '
           'Original methods remain frozen; development gains do not prove L1 superiority. ' if optimization else '')
        + '2026 is evaluated only after the specification is frozen, and its already '
          'reviewed history is not a new blind test. Coefficient sparsity, prediction '
          'quality and net portfolio returns are different measurements.')

    pages.append([
        *_ppt_header("Conclusion", "Historical evidence, with clear limits on persistence and implementation"),
        _ppt_text(f"1   Full net: Lasso-3y {_ppt_percent(_ppt_metric(ctx, 'lasso', 'full'))}; Ridge-exp {_ppt_percent(_ppt_metric(ctx, 'ridge', 'full'))}.", 1.00, 2.03, 11.15, .80, 23, True),
        _ppt_text(f"2   Development: Lasso-3y {_ppt_percent(lasso_dev)}, Ridge-exp {_ppt_percent(ridge_dev)}; {latest}: {_ppt_percent(lasso_latest)} vs {_ppt_percent(ridge_latest)}.", 1.00, 3.33, 11.15, 1.00, 23),
        _ppt_text("3   Supplier revisions, event timing and trading assumptions limit the interpretation.", 1.00, 4.78, 11.15, .95, 23),
        _ppt_text("Questions", 1.00, 6.34, 11.15, .45, 22, True, _ppt_teal),
    ])
    notes.append("Conclude with the actual development comparison and the modest, mixed latest-year evidence. Preserve the unrevised-vintage caveat, synthetic-dividend approximation, company-event assumptions, omitted board lots, minimum commissions and market impact. The early history is fitted replay, and repeated development comparisons can overfit. Invite questions and make sure every group member can explain their assigned part.")

    # Untimed appendices preserve the secondary diagnostics at readable sizes.
    pages.append([
        *_ppt_header("Appendix A | Factor overlap", "Same-date Spearman correlations of the actual annual model inputs"),
        _ppt_chart(charts, "factor_corr", (.85, 1.75, 6.65, 5.05)),
        _ppt_text(f"{len(summary['selected_features'])} used inputs\n\n2021–2025\n\nEvery fifth session\n\nHistorical common pool", 8.05, 2.20, 4.2, 3.00, 20),
        _ppt_text("Correlation describes signal overlap.", 8.05, 5.88, 4.2, .65, 20),
    ])
    notes.append("Appendix for questions; no main-presentation time allocated. The matrix covers the displayed union of used annual model inputs, averaging same-date cross-sectional Spearman matrices over the sampled development dates. It is not a covariance matrix of strategy returns and not an independent test of alpha.")

    positions = ctx["positions"]
    caps = positions[(positions["date"] == positions["date"].max()) & (positions["category"] == "cap")]
    q3 = caps[caps["bucket"].str.startswith("Q3")]
    ridge_q3 = float(q3.loc[q3["method"] == "ridge", "weight"].iloc[0])
    pool_q3 = float(q3.loc[q3["method"] == "benchmark", "weight"].iloc[0])
    pages.append(_ppt_plot_page(charts, "cap_weights", "Appendix B | Actual capital weights",
        f"{end} · Net-NAV-normalized holdings, cash and claims",
        f"Middle-cap Q3: Ridge {ridge_q3:.1%}, pool {pool_q3:.1%}; held capital remains style-exposed."))
    notes.append("Appendix for questions; no main-presentation time allocated. These are actual final held-capital weights normalized by net NAV, including cash and claims, rather than ideal target weights. A score residualized against industry and log capitalization does not force the executed portfolio to be capital neutral. Explain the visible Q3 overweight and the historical-universe quintile construction.")

    pages.append([
        *_ppt_header("Appendix C | Rolling performance", "252-session windows; complete observed history"),
        _ppt_text("NET EXCESS (pp)", .98, 1.78, 5.25, .30, 17, True),
        _ppt_text("ANNUALIZED VOLATILITY (%)", 7.12, 1.78, 5.25, .30, 17, True),
        _ppt_chart(charts, "rolling_excess", (.85, 2.08, 5.5, 4.15)),
        _ppt_chart(charts, "rolling_vol", (6.98, 2.08, 5.5, 4.15)),
        _ppt_text("Rolling observations overlap; initial incomplete windows stay missing.", .98, 6.55, 11.35, .45, 18, color=_ppt_muted),
    ])
    notes.append("Appendix for questions; no main-presentation time allocated. Rolling excess sums the preceding 252 daily matched-cost differences, while rolling volatility annualizes daily net-return dispersion. Windows overlap and are not independent tests. Discuss weak windows as well as strong ones; no new parameter is selected from these charts.")

    if len(pages) != 23 or len(notes) != 23 or sum(_ppt_seconds[:20]) != 1080 or any(_ppt_seconds[20:]):
        raise ValueError("The deck requires twenty timed slides and three untimed appendices")
    presentation = _ppt_Presentation()
    presentation.slide_width = _ppt_Inches(_ppt_width)
    presentation.slide_height = _ppt_Inches(_ppt_height)
    presentation.core_properties.title = "Investment Group 9 | A-share stock selection"
    presentation.core_properties.subject = "Fixed strategy and complete historical evidence"
    presentation.core_properties.author = "Group 9"
    base = _ppt_BytesIO()
    canvas = _ppt_Canvas(base, pagesize=(_ppt_width * 72, _ppt_height * 72))
    canvas.setTitle("Investment Group 9 | Presentation")
    canvas.setAuthor("Group 9")
    placements, geometry = [], []
    text_boxes = tables = figures = 0
    for index, (elements, note, seconds) in enumerate(zip(pages, notes, _ppt_seconds)):
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = _ppt_RGBColor(255, 255, 255)
        canvas.setFillColor(_ppt_colors.white)
        canvas.rect(0, 0, _ppt_width * 72, _ppt_height * 72, fill=1, stroke=0)
        page_label = f"{index + 1:02d}" if index < 20 else f"A{index - 19}"
        elements = [*elements, _ppt_text(page_label, 12.12, 7.14, .45, .20, 10, color=_ppt_muted, align="right")]
        for item in elements:
            if not (item["x"] >= 0 and item["y"] >= 0 and item["x"] + item["w"] <= _ppt_width + .001
                    and item["y"] + item["h"] <= _ppt_height + .001):
                raise ValueError(f"Slide element outside page {index + 1}: {item}")
            if item["type"] == "rule":
                shape = slide.shapes.add_shape(_ppt_MSO_SHAPE.RECTANGLE,
                    _ppt_Inches(item["x"]), _ppt_Inches(item["y"]), _ppt_Inches(item["w"]), _ppt_Inches(item["h"]))
                shape.fill.solid()
                shape.fill.fore_color.rgb = _ppt_RGBColor.from_string(item["color"][1:])
                shape.line.fill.background()
                canvas.setFillColor(_ppt_colors.HexColor(item["color"]))
                canvas.rect(item["x"] * 72, (_ppt_height - item["y"] - item["h"]) * 72,
                            item["w"] * 72, item["h"] * 72, fill=1, stroke=0)
            elif item["type"] == "chart":
                slide.shapes.add_picture(item["png"], _ppt_Inches(item["x"]), _ppt_Inches(item["y"]),
                                        width=_ppt_Inches(item["w"]), height=_ppt_Inches(item["h"]))
                placements.append((index, item))
                figures += 1
            elif item["type"] == "table":
                _ppt_draw_table(slide, canvas, item)
                tables += 1
            else:
                item["line_count"] = _ppt_draw_text(slide, canvas, item)
                text_boxes += 1
            geometry.append({"page": index + 1, **item})
        prefix = f"Suggested speaking time: {seconds} seconds. "
        slide.notes_slide.notes_text_frame.text = prefix + note
        canvas.showPage()
    canvas.save()
    pptx_path, pdf_path = project / "Presentation_Group9.pptx", project / "Presentation_Group9.pdf"
    presentation.save(pptx_path)
    with _ppt_fitz.open(stream=base.getvalue(), filetype="pdf") as document:
        for index, item in placements:
            with _ppt_fitz.open(item["pdf"]) as source:
                rectangle = _ppt_fitz.Rect(item["x"] * 72, item["y"] * 72,
                    (item["x"] + item["w"]) * 72, (item["y"] + item["h"]) * 72)
                document[index].show_pdf_page(rectangle, source, 0)
        document.save(pdf_path, garbage=4, deflate=True)
    return {"pptx": pptx_path, "pdf": pdf_path, "slides": len(pages), "main_slides": 20,
            "appendix_slides": 3, "speaking_seconds": sum(_ppt_seconds), "slide_seconds": list(_ppt_seconds),
            "editable_text_boxes": text_boxes, "native_tables": tables, "separate_chart_images": figures,
            "same_element_geometry": True, "pdf_source": "shared geometry, not an Office export",
            "geometry": geometry}


def build_report(data, config, output_root, project):
    """Regenerate current school deliverables; keep growing assets external."""
    output_root, project = Path(output_root), Path(project)
    ctx = analyze_results(data, config, output_root)
    ctx['positions'] = _position_diagnostics(data, config, output_root)
    with (output_root / 'report_assets/context.pkl').open('wb') as stream:
        _analysis_pickle.dump(ctx, stream, protocol=4)
    charts = make_charts(ctx, output_root)
    report = write_report(ctx, charts, project, output_root)
    presentation = write_presentation(ctx, charts, project, output_root)
    result = {'report': report, 'presentation': presentation,
              'reporting_only': True,
              'diagnostic_label_last_signal': ctx['summary']['last_label_signal']}
    (output_root / 'report_assets/build_manifest.json').write_text(
        json.dumps(result, indent=2, default=str) + '\n')
    print(f"V2 report: {report['body_pages']} body pages; slides: {presentation['slides']}; "
          f"speaking time: {presentation['speaking_seconds']} seconds", flush=True)
    return result
