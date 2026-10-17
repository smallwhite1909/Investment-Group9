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


_analysis_methods = ('fundamental', 'combined', 'ridge', 'lightgbm')
_analysis_all_methods = (*_analysis_methods, 'benchmark')


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
    ctx = {'summary': summary, 'config': config, 'metrics': metrics, 'raw_metrics': raw_metrics,
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
                'monthly_turnover': monthly_turnover, 'cost_components': cost_components}
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


_CHART_METHODS=['fundamental','combined','ridge','lightgbm','benchmark']
_CHART_LABELS={'fundamental':'Fund.','combined':'Combined','ridge':'Ridge','lightgbm':'LGBM','benchmark':'EW pool'}
_CHART_COLORS={'fundamental':'#9063B4','combined':'#D79538','ridge':'#137F82','lightgbm':'#3475AB','benchmark':'#8795A6'}
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


def _make_chart(ctx,key,path,width,height,size):
    """Draw at its final physical dimensions so labels retain their point size."""
    fig=Figure(figsize=(width,height),facecolor='white')
    FigureCanvasAgg(fig)
    is_report=size==12
    compact=height<2.5
    left=.13 if width<4.5 else .095
    bottom=.28 if compact else .20
    top=.75 if compact else .80
    ax=fig.add_axes([left,bottom,.975-left,top-bottom])
    handles=[]
    if key in ['gross_absolute','net_absolute','gross_excess','net_excess']:
        fee=key.split('_')[0]
        returns=ctx['returns'][fee]
        excess=key.endswith('excess')
        curves=returns.sub(returns.benchmark,axis=0).cumsum()*100 if excess else ((1+returns).cumprod()-1)*100
        for method in _CHART_METHODS:
            if excess and method=='benchmark':continue
            ax.plot(curves.index,curves[method],color=_CHART_COLORS[method],label=_CHART_LABELS[method],
                    lw=2.3 if method=='ridge' else 1.3,ls=(0,(4,2)) if method=='benchmark' else '-')
        if excess:
            ax.axhline(0,color=_CHART_COLORS['benchmark'],lw=.8,label='EW pool (0)')
            ax.set_ylabel('Excess (pp)',fontsize=size,color=_CHART_MUTED)
        else:ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0))
        ax.yaxis.set_major_locator(MaxNLocator(3 if compact else 4))
        ax.set_xlim(curves.index[0],curves.index[-1])
        handles,labels=ax.get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.995),ncol=len(labels),frameon=False,
                   fontsize=size,handlelength=1.2,columnspacing=.65,handletextpad=.45)
    elif key=='drawdown':
        frame=ctx['drawdown']
        ax.fill_between(frame.index,frame.ridge.to_numpy()*100,0,color=_CHART_COLORS['ridge'],alpha=.12,linewidth=0)
        for method in ['ridge','benchmark']:
            ax.plot(frame.index,frame[method]*100,color=_CHART_COLORS[method],lw=1.5,
                    label=_CHART_LABELS[method],ls=(0,(4,2)) if method=='benchmark' else '-')
        ax.set_ylim(-42,1);ax.set_yticks([-40,-20,0] if compact else [-40,-30,-20,-10,0])
        ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0))
        handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.995),ncol=2,frameon=False,fontsize=size)
    elif key=='risk_return':
        ax.set_position([.28,.22,.69,.58] if width<4.5 else [.12,.25,.85,.52])
        offsets={'fundamental':(-42,-20),'combined':(7,0),'ridge':(-25,12),'lightgbm':(7,0),'benchmark':(-30,-20)}
        points=[]
        for method in _CHART_METHODS:
            row=_chart_metric(ctx,'rebalance20','development',method)
            x,y=row.annualized_volatility*100,row.annualized_return*100
            ax.scatter(x,y,s=55 if method=='ridge' else 38,color=_CHART_COLORS[method],zorder=4)
            ax.annotate(_CHART_LABELS[method],(x,y),xytext=offsets[method],textcoords='offset points',fontsize=size,color=_CHART_COLORS[method])
            points.append((x,y))
        xy=np.asarray(points)
        ax.set_xlim(xy[:,0].min()-4,xy[:,0].max()+5);ax.set_ylim(xy[:,1].min()-6,xy[:,1].max()+6)
        ax.set_xlabel('Annualized volatility (%)',fontsize=size,color=_CHART_MUTED)
        ax.set_ylabel('Ann. net return (%)',fontsize=size,color=_CHART_MUTED,labelpad=5)
        ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(4))
        fig.text(.55,.93,'2021–2025',ha='center',fontsize=size,color=_CHART_MUTED)
    elif key in ['calendar_excess','calendar_return','factor_corr']:
        fig.delaxes(ax)
        corr=key=='factor_corr'
        calendar=ctx['factor_corr'] if corr else ctx['calendar_excess'] if key=='calendar_excess' else ctx['calendar_return']*100
        values=calendar.to_numpy(dtype=float)
        if corr:values=np.where(np.tril(np.ones_like(values,dtype=bool)),values,np.nan)
        ax=fig.add_axes([.10,.18,.88,.70] if corr else [.085,.19,.90,.67])
        cmap=LinearSegmentedColormap.from_list('report_diverge',['#C46A58','#F7F8F9','#137F82']);cmap.set_bad('#EDF1F4')
        bound=1 if corr else max(1,float(np.ceil(np.nanmax(np.abs(values)))))
        im=ax.imshow(values,cmap=cmap,norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),aspect='auto')
        ax.set_xticks(np.arange(values.shape[1]),list(calendar.columns) if corr else ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],fontsize=size)
        ax.set_yticks(np.arange(values.shape[0]),list(calendar.index),fontsize=size)
        ax.xaxis.tick_top();ax.tick_params(axis='both',length=0,pad=4,colors=_CHART_INK)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value=values[i,j]
                text='' if corr and np.isnan(value) else '—' if np.isnan(value) else f'{value:+.1f}' if abs(value)>=.05 else '0.0'
                color='#9BA7B5' if np.isnan(value) else 'white' if abs(value)>bound*.65 else _CHART_INK
                ax.text(j,i,text,ha='center',va='center',fontsize=size,color=color)
        ax.set_xticks(np.arange(-.5,values.shape[1]),minor=True);ax.set_yticks(np.arange(-.5,values.shape[0]),minor=True)
        ax.grid(which='minor',color='white',linewidth=1);ax.tick_params(which='minor',length=0)
        for spine in ax.spines.values():spine.set_visible(False)
        cax=fig.add_axes([.10,.065,.20,.025]);bar=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-bound,0,bound])
        bar.outline.set_visible(False);bar.ax.tick_params(labelsize=size,length=0,pad=3)
        fig.text(.35,.04,'Mean within-date Spearman' if corr else 'Monthly daily-excess sum (pp)' if key=='calendar_excess' else 'Compounded monthly return (%)',fontsize=size,color=_CHART_MUTED)
    elif key=='annual_return':
        frame=ctx['annual_returns']['net'];x=np.arange(len(frame))
        ax.bar(x-.19,frame.ridge*100,.34,color=_CHART_COLORS['ridge'],label='Ridge',zorder=3)
        ax.bar(x+.19,frame.benchmark*100,.34,color='#A8B3C0',label='EW pool',zorder=3)
        step=2 if width<4.5 else 1
        ax.set_xticks(x[::step],[str(y)[2:] if y<2026 else '26*' for y in frame.index[::step]])
        ax.set_ylim(-25,78);ax.yaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0))
        handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.995),ncol=2,frameon=False,fontsize=size)
    elif key in ['rolling_excess','rolling_vol','turnover','costs','coverage']:
        if key=='rolling_excess':
            frame=ctx['rolling_excess']['ridge'];ax.plot(frame.index,frame,color=_CHART_COLORS['ridge'],lw=1.7)
            ax.axhline(0,color=_CHART_COLORS['benchmark'],lw=.8)
            ax.set_ylabel('Excess (pp)',fontsize=size,color=_CHART_MUTED)
        elif key in ['rolling_vol','turnover']:
            frame=ctx['rolling_vol'] if key=='rolling_vol' else ctx['monthly_turnover']
            for method in ['ridge','benchmark']:
                ax.plot(frame.index,frame[method],color=_CHART_COLORS[method],lw=1.6,label=_CHART_LABELS[method],ls=(0,(4,2)) if method=='benchmark' else '-')
            ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0))
            handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.995),ncol=2,frameon=False,fontsize=size)
        elif key=='costs':
            frame=ctx['cost_components'];base=np.zeros(len(frame))
            for col,color,label in zip(['commission','stamp','slippage'],['#3475AB','#D79538','#137F82'],['Commission','Stamp tax','Slippage']):
                ax.bar(frame.index,frame[col],width=24,bottom=base,color=color,label=label,zorder=3)
                base+=frame[col].to_numpy()
            handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.995),ncol=3,frameon=False,fontsize=size)
        else:
            frame=ctx['coverage']
            ax.plot(frame.date,frame.base_count,color=_CHART_COLORS['benchmark'],label='Historical base pool',lw=1.5)
            ax.plot(frame.date,frame.common_count,color=_CHART_COLORS['ridge'],label='Complete-input pool',lw=1.8)
            handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.995),ncol=2,frameon=False,fontsize=size)
        ax.yaxis.set_major_locator(MaxNLocator(3 if compact else 4))
    elif key=='rank_ic':
        table=ctx['ic_annual'];table=table[table.year.ge(2021)]
        frame=table.pivot(index='year',columns='method',values='mean_ic')
        x=np.arange(len(frame));barwidth=.18
        for k,method in enumerate(_CHART_METHODS[:-1]):
            ax.bar(x+(k-1.5)*barwidth,frame[method],barwidth,color=_CHART_COLORS[method],label=_CHART_LABELS[method],zorder=3)
        ax.axhline(0,color=_CHART_COLORS['benchmark'],lw=.7)
        ax.set_xticks(x,[str(y)[2:] for y in frame.index])
        ax.yaxis.set_major_locator(MaxNLocator(4))
        handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.995),ncol=2 if width<4.5 else 4,frameon=False,fontsize=size,
                                                               handlelength=1.2,columnspacing=.7,handletextpad=.4)
        if width<4.5:ax.set_position([.22,.18,.75,.57])
    elif key in ['quintile','quintile_2026']:
        ax.set_position([.16,bottom,.815,top-bottom])
        period='2021–2025' if key=='quintile' else '2026'
        table=ctx['quintiles'];table=table[table.period.eq(period)]
        for method in _CHART_METHODS[:-1]:
            rows=table[table.method.eq(method)].sort_values('group')
            ax.plot(rows.group,rows.mean_excess*100,color=_CHART_COLORS[method],lw=1.8 if method=='ridge' else 1.25,marker='o',ms=4,label=_CHART_LABELS[method])
        ax.axhline(0,color=_CHART_COLORS['benchmark'],lw=.7)
        ax.set_xticks([1,2,3,4,5],['Q1','Q2','Q3','Q4','Q5'])
        ax.set_ylabel('Mean 20d excess (pp)',fontsize=size,color=_CHART_MUTED)
        ax.yaxis.set_major_locator(MaxNLocator(3 if compact else 4))
        handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.995),ncol=4,frameon=False,fontsize=size,
                                                               handlelength=1.1,columnspacing=.65,handletextpad=.4)
    elif key=='controls':
        fig.delaxes(ax)
        ax1=fig.add_axes([.12,.55,.85,.24]);ax2=fig.add_axes([.12,.12,.85,.23])
        x=np.arange(4)
        raw=[_chart_metric(ctx,'rebalance20','development',method,raw=True).total_return*100 for method in _CHART_METHODS[:-1]]
        adopted=[_chart_metric(ctx,'rebalance20','development',method).total_return*100 for method in _CHART_METHODS[:-1]]
        ax1.bar(x-.19,raw,.34,color='#A8B3C0',label='Raw signals',zorder=3)
        ax1.bar(x+.19,adopted,.34,color=_CHART_COLORS['ridge'],label='Adopted',zorder=3)
        ax1.set_xticks(x,[_CHART_LABELS[m] for m in _CHART_METHODS[:-1]])
        ax1.yaxis.set_major_formatter(PercentFormatter(100,decimals=0));ax1.yaxis.set_major_locator(MaxNLocator(3))
        ax1.set_title('2021–2025 net return · identical 20-session cadence',fontsize=size,loc='left',pad=6,color=_CHART_INK)
        for k,scope in enumerate(['daily','rebalance20','double_slippage']):
            v=_chart_metric(ctx,scope,'holdout','ridge').cumulative_excess*100
            ax2.bar(k,v,.58,color=_CHART_COLORS['ridge'] if k==1 else '#A8B3C0',zorder=3)
        ax2.set_xticks([0,1,2],['Daily','Every 20','Daily, 2× slip'])
        ax2.axhline(0,color=_CHART_COLORS['benchmark'],lw=.7);ax2.yaxis.set_major_locator(MaxNLocator(3))
        ax2.set_title('2026 YTD Ridge · daily net excess sum (pp)',fontsize=size,loc='left',pad=6,color=_CHART_INK)
        handles,labels=ax1.get_legend_handles_labels();fig.legend(handles,labels,loc='upper right',bbox_to_anchor=(.98,.995),ncol=2,frameon=False,fontsize=size)
        for child in [ax1,ax2]:
            child.grid(axis='y',color=_CHART_GRID,lw=.7);child.set_axisbelow(True)
            child.spines[['top','right','left']].set_visible(False);child.spines['bottom'].set_color(_CHART_GRID)
            child.tick_params(axis='both',labelsize=size,length=0,pad=4,colors=_CHART_MUTED)
    elif key=='cap_weights':
        table=ctx['positions'];last=table.date.max()
        table=table[table.date.eq(last)&table.category.eq('cap')]
        frame=table.pivot(index='bucket',columns='method',values='weight').reindex(['Q1 (small)','Q2','Q3','Q4','Q5 (large)','Unknown','Cash','Claims'])
        frame.loc['Other']=frame.loc[['Unknown','Cash','Claims']].sum();frame=frame.iloc[[0,1,2,3,4,8]]
        fig.delaxes(ax);ax=fig.add_axes([.30,.18,.66,.70] if width<4.5 else [.22,.18,.74,.70])
        y=np.arange(len(frame))
        ax.barh(y-.18,frame.ridge*100,.33,color=_CHART_COLORS['ridge'],label='Ridge',zorder=3)
        ax.barh(y+.18,frame.benchmark*100,.33,color='#A8B3C0',label='EW pool',zorder=3)
        ax.set_yticks(y,['Small Q1','Q2','Q3','Q4','Large Q5','Other']);ax.invert_yaxis()
        ax.xaxis.set_major_formatter(PercentFormatter(100,decimals=0));ax.xaxis.set_major_locator(MaxNLocator(3));ax.grid(axis='x',color=_CHART_GRID,lw=.7)
        handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.57,.995),ncol=2,frameon=False,fontsize=size)
    else:
        raise KeyError(key)
    if key not in ['calendar_excess','calendar_return','factor_corr','controls']:
        ax.grid(axis='y' if key!='cap_weights' else 'x',color=_CHART_GRID,lw=.7);ax.set_axisbelow(True)
        ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_color(_CHART_GRID)
        ax.tick_params(axis='both',labelsize=size,length=0,pad=5,colors=_CHART_MUTED)
        if key in ['gross_absolute','net_absolute','gross_excess','net_excess','drawdown','rolling_excess','rolling_vol','turnover','costs','coverage']:
            ax.set_xlim(pd.Timestamp(ctx['summary']['start']),pd.Timestamp(ctx['summary']['end']))
            ax.set_xticks(pd.to_datetime([ctx['summary']['start'],'2020-01-01','2022-01-01','2024-01-01','2026-01-01']))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    path=Path(path)
    fig.savefig(path.with_suffix('.pdf'),metadata={'Title':f'Group 9 | {key}'})
    fig.savefig(path.with_suffix('.png'),dpi=220)
    return {'pdf':str(path.with_suffix('.pdf')),'png':str(path.with_suffix('.png')),'width':width,'height':height}


def make_charts(ctx,output_root):
    """Export slide and A4 variants at their final sizes into external assets."""
    root=Path(output_root)/'report_assets'
    folder=root/'charts';folder.mkdir(parents=True,exist_ok=True)
    report_width=595.275590551/72-2
    dimensions={
      'gross_absolute':((12.15,2.0),(report_width,2.25)),
      'net_absolute':((12.15,2.0),(report_width,2.25)),
      'gross_excess':((12.15,2.0),(report_width,2.25)),
      'net_excess':((12.15,2.0),(report_width,2.25)),
      'drawdown':((8.0,3.7),(report_width,2.0)),
      'risk_return':((3.8,3.7),(report_width,2.0)),
      'calendar_excess':((8.0,3.65),(report_width,2.65)),
      'annual_return':((3.8,3.65),(report_width,2.25)),
      'rolling_excess':((12.15,2.0),(report_width,2.25)),
      'rolling_vol':((12.15,2.0),(report_width,2.25)),
      'turnover':((12.15,2.0),(report_width,2.25)),
      'costs':((12.15,2.0),(report_width,2.25)),
      'coverage':((12.15,3.55),(report_width,2.25)),
      'factor_corr':((8.0,4.0),(report_width,3.3)),
      'rank_ic':((3.8,4.0),(report_width,2.0)),
      'quintile':((5.9,3.6),(report_width,2.25)),
      'quintile_2026':((5.9,3.6),(report_width,2.25)),
      'controls':((8.0,3.7),(report_width,3.3)),
      'cap_weights':((3.8,3.4),(report_width,2.1)),
    }
    charts={}
    with matplotlib.rc_context({'font.family':'DejaVu Sans','pdf.fonttype':42,'axes.unicode_minus':False}):
        for key,pair in dimensions.items():
            charts[key]={}
            for mode,(width,height) in zip(['slide','report'],pair):
                size=12 if mode=='report' else 16 if key in ['coverage','drawdown'] else 14
                charts[key][mode]=_make_chart(ctx,key,folder/f'{key}_{mode}',width,height,size)
    # Retain the exact four-curve numerical contract, independent of layout.
    rows=[]
    for method in _CHART_METHODS:
        gross,net=ctx['returns']['gross'],ctx['returns']['net']
        rows.append(pd.DataFrame({'date':gross.index,'method':method,
            'gross_absolute':(1+gross[method]).cumprod().to_numpy()-1,
            'net_absolute':(1+net[method]).cumprod().to_numpy()-1,
            'gross_excess_pp':(gross[method]-gross.benchmark).cumsum().to_numpy()*100,
            'net_excess_pp':(net[method]-net.benchmark).cumsum().to_numpy()*100}))
    pd.concat(rows,ignore_index=True).to_csv(root/'tables/FourCharts_DailyValues.csv',index=False,float_format='%.12g')
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
_report_methods = ['fundamental', 'combined', 'ridge', 'lightgbm', 'benchmark']
_report_names = {
    'fundamental': 'Fundamental', 'combined': 'Combined', 'ridge': 'Ridge',
    'lightgbm': 'LightGBM', 'benchmark': 'Benchmark',
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


def write_report(ctx, charts, project, output_root):
    """Write the current 14-page results report plus one reference page.

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
    end_date = _report_date(summary['end_date'])
    first_trade = _report_date(summary['first_trade'])
    label_last = _report_date(summary['label_last_signal'])
    specs = config['method_specs']
    ridge_params = specs['ridge']['model_params']
    tree_params = specs['lightgbm']['model_params']
    model_settings = config['models']

    # Page 1: executive summary and an immediate, matched performance comparison.
    summary_rows = [['Portfolio', 'Net return', 'Max. drawdown', 'Net excess (pp)']]
    for method in _report_methods:
        row = net[method]
        summary_rows.append([
            _report_names[method], _report_percent(row['total_return']),
            _report_percent(row['max_drawdown']), _report_pp(row['cumulative_excess'], 1),
        ])
    pages = [[
        _report_paragraph('Fundamental and technical signals for A-share stock selection', 'heading'),
        ('gap', 10),
        _report_paragraph('Investments 1005 | Group 9', 'subheading'),
        _report_paragraph(
            f'Historical daily data: {_report_date(summary["start_date"])} to {end_date}. '
            f'The main implementation is a long-only, equal-weight Top-{config["top_n"]} '
            f'portfolio with targets refreshed every {config["rebalance_sessions"]} market sessions.'),
        ('gap', 10),
        _report_paragraph('Research question', 'subheading'),
        _report_paragraph(
            'Can accounting and technical information improve stock selection after trading costs '
            'within the same opportunity set? We compare a fundamental score, a combined score, '
            'Ridge regression and a shallow LightGBM model. A matched equal-weight portfolio '
            'isolates stock-selection performance from differences in the available stock pool.'),
        ('gap', 10),
        _report_paragraph('Main findings', 'subheading'),
        _report_paragraph(
            f'Ridge earns {_report_percent(ridge["total_return"])} net over the complete history, '
            f'versus {_report_percent(benchmark["total_return"])} for the benchmark. Its maximum '
            f'drawdown is {_report_percent(abs(ridge["max_drawdown"]))}, compared with '
            f'{_report_percent(abs(benchmark["max_drawdown"]))}. Higher historical return therefore '
            'does not imply lower downside risk.'),
        ('gap', 8),
        _report_table(summary_rows, [86, 90, 126, width - 302]),
        ('gap', 10),
        _report_paragraph(
            f'For 2021–2025, Ridge returns {_report_percent(development["ridge"]["total_return"])} '
            f'net, versus {_report_percent(development["benchmark"]["total_return"])} for the '
            f'benchmark. In 2026 through {end_date}, the corresponding returns are '
            f'{_report_percent(recent["ridge"]["total_return"], 2)} and '
            f'{_report_percent(recent["benchmark"]["total_return"], 2)}; cumulative daily net '
            f'excess is {_report_pp(recent["ridge"]["cumulative_excess"])} percentage points.'),
        ('gap', 8),
        _report_paragraph(
            'All returns in the table use the same cadence and cost assumptions. Absolute returns '
            'compound daily wealth; excess sums daily strategy-minus-benchmark returns. These '
            'different definitions must not be interpreted as the same terminal-wealth difference. '
            'The partial 2026 record and historical implementation assumptions limit the strength '
            'of any claim about future performance.'),
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
            'determine whether a stock can be selected.'), ('gap', 8),
        _report_paragraph(
            'Financial observations require report periods and both recorded announcement dates '
            'no later than the signal day. These guards reduce date inconsistencies but cannot '
            'certify the absence of later vendor revisions. Complete-input filtering also favors '
            'firms with a longer and more complete reporting history. Initial cash while lagged '
            f'features accumulate explains the flat beginning before {first_trade}.'),
    ])

    # Page 4: the one explicit fitted-history statement and actual fixed settings.
    model_rows = [['Method', 'Construction and fixed settings']]
    for method in _report_methods[:-1]:
        spec = specs[method]
        span = int(spec['processing']['prediction_rank_ema_span'])
        feature_text = ', '.join(spec['features'])
        if method == 'fundamental':
            construction = 'Mean signed financial ranks.'
        elif method == 'combined':
            construction = 'Financial and technical family scores receive equal weight.'
        elif method == 'ridge':
            construction = f'L2-regularized regression; alpha={ridge_params["alpha"]:g}; one BLAS thread. [1]'
        else:
            construction = (
                f'Depth {tree_params["max_depth"]}; {tree_params["num_leaves"]} leaves; '
                f'{tree_params["boosting_rounds"]} trees; minimum leaf '
                f'{tree_params["min_child_samples"]}; learning rate '
                f'{model_settings["learning_rate"]:g}. [2]')
        model_rows.append([
            _report_names[method], f'{construction} Score EMA span {span}. Input union: {feature_text}.',
        ])
    pages.append([
        _report_paragraph('3. Forecasts, score processing and fixed models', 'heading'), ('gap', 10),
        _report_paragraph(
            'Each daily cross section is converted to centered average ranks. The prediction '
            f'target is the adjusted opening return from t+1 to t+{config["horizon"] + 1}, '
            'demeaned across eligible labeled stocks. Ridge provides a regularized linear '
            'comparison; a shallow tree model permits nonlinear interactions.'), ('gap', 8),
        _report_table(model_rows, [88, width - 88]), ('gap', 10),
        _report_paragraph('Causal score processing', 'subheading'),
        _report_paragraph(
            'Prediction ranks are residualized against industry intercepts and within-industry '
            'log market capitalization, then reranked. A causal score EMA follows, with '
            'alpha=2/(span+1), followed by daily reranking. First finite values initialize state; '
            'missing inputs emit no score and preserve the previous state. State continues '
            'across year boundaries. Input EMA and the extra fundamental-score blend are zero.'),
        ('gap', 8),
        _report_paragraph('Training chronology', 'subheading'),
        _report_paragraph(
            f'Annual refits use expanding prior history sampled every '
            f'{model_settings["training_stride"]} market sessions. A training observation is '
            'admitted only when its t+21 exit is strictly earlier than the prediction start. '
            'The 2018–2020 portion is a fitted replay using the pre-2021 model; from 2021, '
            'each annual model uses mature earlier labels. The full curve is therefore a '
            'historical backtest, not a claim that every point was an independent forecast.'),
        ('gap', 8),
        _report_paragraph(
            'Annual input lists can differ from their displayed unions. Model and processing '
            'choices used daily 2021–2025 development comparisons; the displayed portfolios '
            'replay the adopted 20-session implementation. Score residualization does not '
            'guarantee industry-neutral holdings or reduced drawdown.'),
    ])

    # Page 5: match both execution and costs, including event-accounting limits.
    cost_rows = [
        ['Cost or execution item', 'Assumption'],
        ['Commission, each side', f'{config["commission"] * 10000:g} basis points'],
        ['Slippage, each side', f'{config["slippage"] * 10000:g} basis points'],
        ['Sell stamp duty',
         f'{config["stamp_before"] * 10000:g} bp before 28 Aug 2023; '
         f'{config["stamp_after"] * 10000:g} bp thereafter [3, 4]'],
        ['Target / execution / valuation', '20-session refresh / next open / daily'],
    ]
    pages.append([
        _report_paragraph('4. Portfolio, benchmark and transaction accounting', 'heading'), ('gap', 10),
        _report_paragraph(
            f'The main portfolio selects the highest-scoring {config["top_n"]} names at equal '
            'target weights. Targets refresh every 20 market sessions on one continuous '
            'calendar. Scores observed after the preceding close determine next-open orders. '
            'Data, score updates and valuation remain daily, and blocked orders can be retried '
            'between target refreshes. Actual positions can differ from the intended Top 100.'),
        ('gap', 8),
        _report_paragraph(
            'The primary benchmark equal-weights the identical historical common pool, with '
            'the same target cadence, opening execution constraints and cost assumptions. '
            'It is an opportunity-set benchmark, not the CSI500 price index. Gross strategies '
            'are compared with the gross benchmark; net strategies with the net benchmark.'),
        ('gap', 8), _report_table(cost_rows, [186, width - 186]), ('gap', 10),
        _report_paragraph(
            'Suspensions, zero volume and opening price limits can block trades. Cash limits '
            'and T+1 apply; an unsold holding remains in the book. Gross and net portfolios '
            'are simulated separately, so fees can also alter later quantities and cash. '
            'Round lots, minimum commissions, small transfer fees and liquidity impact are '
            'omitted. Daily bars do not establish auction queue priority.'), ('gap', 8),
        _report_paragraph('Corporate actions and missing prices', 'subheading'),
        _report_paragraph(
            'A causal close/previous-close chain approximates reinvested corporate actions. '
            'Sourced share exchanges and cash exits are processed with their settlement '
            'assumptions; unavailable claims remain untradeable until release. Quote-less '
            'suspensions carry the last valid mark. The 60-day missing-price threshold is '
            'diagnostic, not an automatic exit. Unresolved formal listing ends assume zero '
            'recovery, which is a valuation convention rather than proof of zero eventual recovery.'),
        ('gap', 8),
        _report_paragraph(
            'The price chain does not reproduce exact cash-dividend receipt dates, dividend '
            'taxes or every rights subscription. These simplifications and uncertain historical '
            'data vintages affect the economic interpretation of simulated returns.'),
    ])

    # Pages 6–7: four complete-history curves, with no stage shading or labels.
    pages.append([
        _report_paragraph('5. Complete-history absolute returns', 'heading'), ('gap', 10),
        _report_paragraph(
            'Absolute curves show portfolio NAV minus one, using the complete daily history. '
            'They retain cash, execution constraints and compounding along each path.'), ('gap', 8),
        *_report_chart('gross_absolute',
            f'Figure 2. Cumulative gross returns, 2 Jan 2018–{end_date}. '
            'Each strategy is compared with the gross common-pool benchmark.'),
        *_report_chart('net_absolute',
            f'Figure 3. Cumulative net returns over the same complete history, including '
            'commission, sell stamp duty and slippage.'),
        _report_paragraph(
            f'Ridge ends at {_report_percent(gross["ridge"]["total_return"])} gross and '
            f'{_report_percent(ridge["total_return"])} net. The benchmark ends at '
            f'{_report_percent(gross["benchmark"]["total_return"])} gross and '
            f'{_report_percent(benchmark["total_return"])} net. The difference between a '
            'portfolio’s gross and net terminal wealth reflects compounding and separately '
            'simulated trading paths; it is not the simple sum of its fee payments.'),
    ])
    pages.append([
        _report_paragraph('6. Complete-history excess returns', 'heading'), ('gap', 10),
        _report_paragraph(
            'Each excess curve accumulates daily strategy returns minus the matching '
            'benchmark return. Units are percentage points. The benchmark is zero by '
            'construction; the measure does not compound a relative-wealth portfolio.'), ('gap', 8),
        *_report_chart('gross_excess',
            'Figure 4. Cumulative daily gross excess versus the same-pool gross benchmark, '
            f'2 Jan 2018–{end_date}.'),
        *_report_chart('net_excess',
            'Figure 5. Cumulative daily net excess versus the same-pool net benchmark, '
            'including the matching execution costs.'),
        _report_paragraph(
            f'Ridge accumulates {_report_pp(ridge["cumulative_excess"])} pp of daily net '
            f'excess over the full history, including '
            f'{_report_pp(development["ridge"]["cumulative_excess"])} pp in 2021–2025 '
            f'and {_report_pp(recent["ridge"]["cumulative_excess"])} pp in 2026 through '
            f'{end_date}. A rising segment indicates positive cumulative selection advantage '
            'over that interval; a falling segment shows benchmark outperformance.'),
    ])

    # Page 8: return is paired with observed risk, not a one-number sales claim.
    risk_rows = [['Portfolio', 'Net CAGR', 'Annual vol.', 'Max. drawdown']]
    for method in _report_methods:
        row = net[method]
        risk_rows.append([
            _report_names[method], _report_percent(row['annualized_return']),
            _report_percent(row['annualized_volatility']), _report_percent(row['max_drawdown']),
        ])
    pages.append([
        _report_paragraph('7. Drawdowns and the return–risk trade-off', 'heading'), ('gap', 10),
        *_report_chart('drawdown',
            'Figure 6. Net-wealth drawdowns relative to each portfolio’s prior peak. '
            'Negative values measure losses from that peak.'),
        *_report_chart('risk_return',
            'Figure 7. Net CAGR versus annualized daily volatility in 2021–2025. '
            'The table below instead records the complete active trading history.'),
        _report_table(risk_rows, [100, 92, 108, width - 300]), ('gap', 8),
        _report_paragraph(
            f'Ridge’s worst drawdown is {_report_percent(abs(ridge["max_drawdown"]))}, '
            f'slightly deeper than the benchmark’s '
            f'{_report_percent(abs(benchmark["max_drawdown"]))}. Its annualized volatility '
            f'is {_report_percent(ridge["annualized_volatility"])}, compared with '
            f'{_report_percent(benchmark["annualized_volatility"])}. The historical return '
            'advantage does not remove market exposure, drawdowns or model risk. '
            'Annualization uses 252 market sessions; no risk-free return is assumed for Sharpe.'),
    ])

    # Page 9: a different aggregation is explicitly distinguished from compounding.
    pages.append([
        _report_paragraph('8. Performance across months and calendar years', 'heading'), ('gap', 10),
        _report_paragraph(
            'Monthly excess and calendar-year returns expose variation hidden by a final '
            'cumulative number. The two views aggregate the same daily observations '
            'using different definitions.'), ('gap', 8),
        *_report_chart('calendar_excess',
            'Figure 8. Ridge monthly net excess in percentage points: the sum of daily '
            'strategy-minus-benchmark returns. Gray dashes denote unobserved months.'),
        *_report_chart('annual_return',
            f'Figure 9. Compounded calendar-year net returns. The 2026 observations '
            f'end on {end_date}; the final bar is a partial year.'),
        _report_paragraph(
            f'The 2021–2025 Ridge net return is '
            f'{_report_percent(development["ridge"]["total_return"])}, while 2026 through '
            f'{end_date} returns {_report_percent(recent["ridge"]["total_return"], 2)}. '
            'Negative monthly excess remains possible within a profitable overall history. '
            'The incomplete 2026 result is not directly comparable with a full-year bar '
            'and does not establish persistence across future market conditions.'),
    ])

    # Page 10: fee bars use their own accounting denominator, not NAV drag.
    daily_ridge = _report_metric(ctx, 'ridge', scope='daily')
    turnover_rows = [
        ['Portfolio', 'Mean daily turnover', 'Annualized turnover'],
        ['Ridge, 20-session targets', _report_percent(ridge['average_daily_turnover'], 2),
         f'{ridge["annualized_turnover"]:.2f}x'],
        ['Benchmark, same targets', _report_percent(benchmark['average_daily_turnover'], 2),
         f'{benchmark["annualized_turnover"]:.2f}x'],
        ['Ridge, daily targets', _report_percent(daily_ridge['average_daily_turnover'], 2),
         f'{daily_ridge["annualized_turnover"]:.2f}x'],
    ]
    pages.append([
        _report_paragraph('9. Turnover and the sources of trading costs', 'heading'), ('gap', 10),
        *_report_chart('turnover',
            'Figure 10. Monthly summed daily turnover for Ridge and the benchmark. '
            'Daily turnover is half the traded notional divided by opening wealth.'),
        *_report_chart('costs',
            'Figure 11. Ridge commission, stamp duty and slippage, scaled by each '
            'transaction day’s pre-trade opening NAV and summed within month (pp).'),
        _report_table(turnover_rows, [178, 142, width - 320]), ('gap', 8),
        _report_paragraph(
            'Twenty-session target updates reduce target replacement relative to the daily '
            'comparison, but blocked-order retries and changing portfolio values still create '
            'trades between updates. A 20-session strategy therefore need not trade on only '
            'one day per month. The component bars measure accounting charges; their sum '
            'must not be read as the gross-minus-net terminal return difference.'),
    ])

    # Page 11: factor dependence and forecast quality, without conflating either with P&L.
    pages.append([
        _report_paragraph('10. Input dependence and forecast rank correlation', 'heading'), ('gap', 10),
        _report_paragraph(
            'Correlation identifies overlapping inputs; rank IC asks whether a higher model '
            'score tends to align with a higher future return across stocks. Neither measure '
            'includes turnover, transaction costs or actual opening fills.'), ('gap', 8),
        *_report_chart('factor_corr',
            'Figure 12. Adopted-input rank correlations: equal-date means of daily '
            'cross-sectional Spearman correlations in 2021–2025, sampled every fifth '
            'session in the common pool.'),
        *_report_chart('rank_ic',
            'Figure 13. Annual mean daily Spearman rank IC of adopted scores against '
            'mature 20-session opening-return labels. The 2021–2025 values require '
            'label exits by 31 Dec 2025; 2026 exits end on 29 Apr 2026.'),
        _report_paragraph(
            f'T3 and T5 have rank correlation '
            f'{ctx["factor_corr"].loc["T3", "T5"]:+.2f}, indicating overlapping volatility '
            f'and activity information. Ridge’s 2026 mean IC is '
            f'{ctx["ic_annual"].loc[ctx["ic_annual"]["year"].eq(2026) & ctx["ic_annual"]["method"].eq("ridge"), "mean_ic"].iloc[0]:+.4f}, '
            'but a positive whole-pool IC can coexist with weak high-score selection. '
            'The next page examines that tail. Overlapping labels make daily IC '
            'observations dependent, so the figure is descriptive.'),
    ])

    # Page 12: forward-label diagnostics are explicitly not tradable bucket backtests.
    spread_rows = [['Method', '2021–2025 Q5-Q1 (pp)', '2026 Q5-Q1 (pp)']]
    quintile_spreads = {}
    for method in _report_methods[:-1]:
        values = []
        quintile_spreads[method] = {}
        for period in ['2021–2025', '2026']:
            grouped = ctx['quintiles'].loc[
                ctx['quintiles']['period'].eq(period) & ctx['quintiles']['method'].eq(method)
            ].set_index('group')
            spread = grouped.loc[5, 'mean_excess'] - grouped.loc[1, 'mean_excess']
            quintile_spreads[method][period] = spread
            values.append(_report_pp(spread))
        spread_rows.append([_report_names[method], *values])
    pages.append([
        _report_paragraph('11. Does the high-score group contain better returns?', 'heading'), ('gap', 10),
        *_report_chart('quintile',
            'Figure 14. Mean future 20-session return excess by score quintile, '
            '2021–2025. Q1 is the lowest score; Q5 is the highest.'),
        *_report_chart('quintile_2026',
            f'Figure 15. The same diagnostic for available 2026 signal dates through '
            f'{label_last}. Later signals do not yet have mature labels.'),
        _report_table(spread_rows, [94, 177, width - 271]), ('gap', 8),
        _report_paragraph(
            'These are future-label cohort averages, not portfolio returns: they ignore '
            'fees, opening execution limits, cash and delayed sales. The comparison uses '
            'near-equal score groups before removing unavailable labels; ties use fixed '
            'security order. Excess subtracts the daily labeled score-pool mean, and '
            'daily cohort means receive equal weight. Labels ending after 2025 are '
            'excluded from the 2021–2025 panel. Adjacent cohorts overlap and are '
            'not independent. Ridge’s Q5-Q1 spread moves from '
            f'{_report_pp(quintile_spreads["ridge"]["2021–2025"])} pp to '
            f'{_report_pp(quintile_spreads["ridge"]["2026"])} pp, weakening high-score '
            'separation in 2026.'),
    ])

    # Page 13: preserve all methods and the correct frequency of each control.
    recent_rows = [['Portfolio', '2026 gross', '2026 net', 'Net excess (pp)']]
    for method in _report_methods:
        recent_rows.append([
            _report_names[method], _report_percent(recent_gross[method]['total_return'], 2),
            _report_percent(recent[method]['total_return'], 2),
            _report_pp(recent[method]['cumulative_excess']),
        ])
    raw_ridge = _report_metric(ctx, 'ridge', 'development', raw=True)
    pages.append([
        _report_paragraph('12. Controls and the 2026 method comparison', 'heading'), ('gap', 10),
        *_report_chart('controls',
            'Figure 16. Above: 2021–2025 net returns for raw and adopted scores, '
            'with the same pool and 20-session schedule. Below: 2026 Ridge daily net '
            'excess under daily targets, 20-session targets and daily doubled slippage.'),
        _report_table(recent_rows, [94, 101, 101, width - 296]), ('gap', 8),
        _report_paragraph(
            f'The raw 20-session Ridge baseline returns '
            f'{_report_percent(raw_ridge["total_return"])} net in 2021–2025, versus '
            f'{_report_percent(development["ridge"]["total_return"])} for the adopted '
            'processed implementation. This comparison changes score processing while '
            'holding the historical pool and cadence fixed.'), ('gap', 8),
        _report_paragraph(
            'The doubled-slippage book is a daily implementation control, not a stress '
            'test of the main 20-session schedule. Comparisons across frequencies change '
            'both the timing of holdings and trading costs. The 2026 table retains every '
            'method. Combined leads this interval at '
            f'{_report_percent(recent["combined"]["total_return"], 2)} net, while Ridge '
            'remains the main model. The short record does not demonstrate a stable '
            'ordering of methods.'),
    ])

    # Page 14: limits and a conclusion that does not overstate evidence or style neutrality.
    final_blocks = [
        _report_paragraph('13. Limitations and conclusions', 'heading'), ('gap', 10),
        _report_paragraph('What the results support', 'subheading'),
        _report_paragraph(
            'The matched historical comparison supports further study of regularized '
            'cross-sectional ranking with causal score smoothing. Ridge combines a '
            'transparent linear model with a common opportunity set and explicit execution '
            'costs. Its stronger historical return is accompanied by meaningful drawdown '
            'and an inconsistent advantage across shorter intervals.'), ('gap', 8),
        _report_paragraph('What remains uncertain', 'subheading'),
        _report_paragraph(
            'Historical specification choices can favor patterns that do not persist. '
            'The reviewed 2026 record is short and cannot establish an untouched prospective '
            'result. Overlapping return labels limit statistical independence. Complete-input '
            'filtering changes the opportunity set, while vendor revisions and announcement '
            'timestamps do not certify original unrevised accounting vintages.'), ('gap', 8),
        _report_paragraph(
            'Execution uses daily opening information and omits market impact, queue '
            'priority, lot rounding and minimum commissions. Corporate-action valuation '
            'and unresolved exit recovery are approximations. Score residualization removes '
            'specific cross-sectional components before ranking; it does not enforce neutral '
            'actual holdings or guarantee lower risk.'), ('gap', 8),
    ]
    if 'cap_weights' in charts:
        final_blocks.extend(_report_chart('cap_weights',
            f'Figure 17. Actual net-NAV weights on {end_date} by market-cap quintile '
            '(Q1 small, Q5 large), using same-day eligible-pool boundaries. Other '
            'combines cash, pending claims and unknown capitalization.'))
    final_blocks.append(_report_paragraph(
        f'The main conclusion is a historically attractive net stock-selection result '
        f'with material implementation and estimation limits. Ridge’s '
        f'{_report_percent(recent["ridge"]["total_return"], 2)} net return in 2026 through '
        f'{end_date}, against {_report_percent(recent["benchmark"]["total_return"], 2)} '
        'for the matched benchmark, is modest evidence over a partial year. '
        'A future prospective run with unchanged rules would provide a stronger check.'))
    pages.append(final_blocks)

    # References are separate from the 14-page main-body budget.
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
        _report_paragraph(
            'The report contains 14 main-body pages, including all figures and tables, '
            'plus this reference page. Body text, table text, chart labels and captions '
            'use 12-point type; body leading is 18 points with one-inch margins. '
            'Report figures are embedded as vector PDFs.'),
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
        'path': str(pdf_path), 'body_pages': 14, 'total_pages': len(pages),
        'body_font_pt': 12, 'body_leading_pt': 18, 'margins_pt': 72,
        'blocks': bounds,
    }


"""Editable Group 9 slides and a PDF drawn from the same element geometry."""

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
from reportlab.pdfgen.canvas import Canvas as _ppt_Canvas
from reportlab.pdfbase import pdfmetrics as _ppt_pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont as _ppt_TTFont
from reportlab.lib import colors as _ppt_colors


_ppt_ink = "#162B42"
_ppt_teal = "#137F82"
_ppt_muted = "#657489"
_ppt_coral = "#BF6254"
_ppt_paper = "#F1F4F7"
_ppt_grid = "#E3E9EF"
_ppt_width = 13.333333
_ppt_height = 7.5
_ppt_methods = ("fundamental", "combined", "ridge", "lightgbm", "benchmark")
_ppt_labels = {"fundamental": "Fundamental", "combined": "Combined", "ridge": "Ridge",
               "lightgbm": "LightGBM", "benchmark": "Benchmark"}
_ppt_seconds = [40, 70, 65, 80, 75, 95, 85, 85, 85, 65, 80, 80, 70, 65, 40]


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


def _ppt_box(x, y, w, h, fill="#FFFFFF"):
    return {"type": "box", "x": x, "y": y, "w": w, "h": h, "fill": fill}


def _ppt_chart(charts, key, x, y, width, height):
    art = charts[key]["slide"]
    if abs(float(art["width"]) - width) > .01 or abs(float(art["height"]) - height) > .01:
        raise ValueError(f"Slide chart {key} needs {width} x {height} inches, not "
                         f"{art['width']} x {art['height']}; export at the intended physical size")
    return {"type": "chart", "key": key, "png": str(art["png"]), "pdf": str(art["pdf"]),
            "x": x, "y": y, "w": width, "h": height}


def _ppt_table(rows, x, y, widths, row_height=.43, size=17, highlight_rows=()):
    return {"type": "table", "rows": rows, "x": x, "y": y, "w": sum(widths),
            "h": len(rows) * row_height, "widths": widths, "row_height": row_height,
            "size": size, "highlight_rows": list(highlight_rows)}


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


def _ppt_header(title, subtitle):
    return [_ppt_text(title, .55, .63, 12.2, .55, 28, True),
            _ppt_text(subtitle, .55, 1.28, 12.2, .42, 14, color=_ppt_muted)]


def _ppt_card(x, y, w, h, label, value, detail="", value_color=_ppt_teal):
    items = [_ppt_box(x, y, w, h),
             _ppt_text(label, x + .18, y + .13, w - .36, .25, 13, True, _ppt_muted),
             _ppt_text(value, x + .18, y + .43, w - .36, .43, 29, True, value_color)]
    if detail:
        items.append(_ppt_text(detail, x + .18, y + .96, w - .36, .45, 14, color=_ppt_muted))
    return items


def _ppt_two_charts(charts, title, subtitle, upper, lower, upper_label, lower_label, takeaway):
    return [*_ppt_header(title, subtitle),
            _ppt_text(upper_label, .65, 1.78, 12, .25, 15, True),
            _ppt_chart(charts, upper, .55, 2.05, 12.15, 2.0),
            _ppt_text(lower_label, .65, 4.18, 12, .25, 15, True),
            _ppt_chart(charts, lower, .55, 4.45, 12.15, 2.0),
            _ppt_text(takeaway, .65, 6.62, 12, .38, 20)]


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


def _ppt_draw_table(slide, canvas, item):
    rows, widths, height, size = item["rows"], item["widths"], item["row_height"], item["size"]
    shape = slide.shapes.add_table(len(rows), len(widths), _ppt_Inches(item["x"]),
                                  _ppt_Inches(item["y"]), _ppt_Inches(item["w"]),
                                  _ppt_Inches(item["h"]))
    table = shape.table
    for column, width in zip(table.columns, widths):
        column.width = _ppt_Inches(width)
    for row in table.rows:
        row.height = _ppt_Inches(height)
    for r, values in enumerate(rows):
        x = item["x"]
        for c, (value, width) in enumerate(zip(values, widths)):
            fill = "#E3EFF0" if r == 0 else "#EAF3F3" if r in item["highlight_rows"] else "#FFFFFF"
            cell = table.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = _ppt_RGBColor.from_string(fill[1:])
            cell.margin_left = cell.margin_right = _ppt_Inches(.08)
            cell.margin_top = _ppt_Inches(.055)
            cell.margin_bottom = 0
            cell.vertical_anchor = _ppt_MSO_ANCHOR.TOP
            frame = cell.text_frame
            frame.clear()
            frame.word_wrap = False
            lines = _ppt_wrap(value, width - .16, size, r == 0)
            if size + (len(lines) - 1) * size * 1.1 > (height - .07) * 72:
                raise ValueError(f"Table cell overflows: {value}")
            p = frame.paragraphs[0]
            p.text = "\v".join(lines)
            p.font.name = "DejaVu Sans"
            p.font.size = _ppt_Pt(size)
            p.font.bold = r == 0
            p.font.color.rgb = _ppt_RGBColor.from_string(_ppt_ink[1:])
            p.space_before = p.space_after = _ppt_Pt(0)
            p.line_spacing = _ppt_Pt(size * 1.1)
            canvas.setFillColor(_ppt_colors.HexColor(fill))
            canvas.rect(x * 72, _ppt_height * 72 - (item["y"] + (r + 1) * height) * 72,
                        width * 72, height * 72, fill=1, stroke=0)
            canvas.setFillColor(_ppt_colors.HexColor(_ppt_ink))
            canvas.setFont("G9SlideBold" if r == 0 else "G9Slide", size)
            for number, line in enumerate(lines):
                canvas.drawString((x + .08) * 72,
                                  _ppt_height * 72 - (item["y"] + r * height + .055) * 72
                                  - size * .86 - number * size * 1.1, line)
            canvas.setStrokeColor(_ppt_colors.HexColor(_ppt_grid))
            canvas.setLineWidth(.5)
            canvas.line(x * 72, _ppt_height * 72 - (item["y"] + (r + 1) * height) * 72,
                        (x + width) * 72, _ppt_height * 72 - (item["y"] + (r + 1) * height) * 72)
            x += width


def write_presentation(ctx, charts, project, output_root):
    """Write 15 editable slides and a matching vector-chart PDF preview.

    Both formats are assembled from the same explicit text/table/chart geometry.
    The PDF is generated here; it is not represented as an Office export.
    Chart PNGs are separate figures in PPTX, never slide-sized screenshots.
    """
    _ppt_register_fonts()
    project = _ppt_Path(project)
    project.mkdir(parents=True, exist_ok=True)
    summary = ctx["summary"]
    end = _ppt_human_date(summary["end"])
    span = f"{_ppt_human_date(summary['start'])} – {end}"
    latest = int(summary["latest_year"])
    ridge_development = _ppt_metric(ctx, "ridge", "development")
    ridge_latest = _ppt_metric(ctx, "ridge", "holdout")
    latest_excess = _ppt_metric(ctx, "ridge", "holdout", "cumulative_excess")
    full_dd = _ppt_metric(ctx, "ridge", "full", "max_drawdown")
    benchmark_dd = _ppt_metric(ctx, "benchmark", "full", "max_drawdown")

    pages = []
    notes = []
    result_rows = [["Method", "2021–2025 net", f"{latest} net", f"{latest} excess (pp)"]]
    for method in _ppt_methods:
        result_rows.append([_ppt_labels[method],
                            _ppt_percent(_ppt_metric(ctx, method, "development")),
                            _ppt_percent(_ppt_metric(ctx, method, "holdout")),
                            f"{_ppt_metric(ctx, method, 'holdout', 'cumulative_excess') * 100:+.2f}"])
    pages.append([
        *_ppt_header("A-share cross-sectional stock selection", f"Four fixed methods  |  Top 100  |  20-session targets  |  {span}"),
        *_ppt_card(.55, 1.92, 3.90, 1.13, "2021–2025 RIDGE NET", _ppt_percent(ridge_development)),
        *_ppt_card(4.72, 1.92, 3.90, 1.13, f"{latest} RIDGE NET", _ppt_percent(ridge_latest)),
        *_ppt_card(8.89, 1.92, 3.90, 1.13, f"{latest} RIDGE NET EXCESS", _ppt_pp(latest_excess)),
        _ppt_table(result_rows, .55, 3.40, [3.20, 3.15, 2.80, 3.00], .45, 17, [3]),
        _ppt_text("Ridge adds net return; the latest-year excess remains modest.", .65, 6.40, 12, .45, 20),
    ])
    notes.append("State the investment question and four methods. Distinguish compounded net returns from the sum of daily matched-benchmark excess. Read the partial-year result without annualizing it; do not imply that every method wins.")

    pages.append([
        *_ppt_header("Economic mechanisms behind the candidate inputs", "Fourteen candidates define the common pool; fixed annual selections determine actual model inputs"),
        _ppt_text("Combine complementary hypotheses, then test their realised behaviour.", .65, 1.82, 12, .44, 20),
        _ppt_box(.55, 2.50, 3.90, 3.55), _ppt_box(4.72, 2.50, 3.90, 3.55), _ppt_box(8.89, 2.50, 3.90, 3.55),
        _ppt_text("PROFITABILITY & CASH", .76, 2.73, 3.48, .34, 18, True, _ppt_teal),
        _ppt_text("ROE and ROA dynamics\nRevenue-growth dynamics\nROA level\nOperating cash / revenue", .76, 3.28, 3.48, 2.10, 20),
        _ppt_text("F1 · F2 · F3 · F8 · F9", .76, 5.57, 3.48, .28, 15, color=_ppt_muted),
        _ppt_text("VALUATION & LEVERAGE", 4.93, 2.73, 3.48, .34, 18, True, _ppt_teal),
        _ppt_text("Signed book / market\nSigned TTM earnings yield\nDebt / equity dynamics\nLow debt / assets", 4.93, 3.28, 3.48, 2.10, 20),
        _ppt_text("F5 · F6 · F7 · F10", 4.93, 5.57, 3.48, .28, 15, color=_ppt_muted),
        _ppt_text("PRICE & TRADING", 9.10, 2.73, 3.48, .34, 18, True, _ppt_teal),
        _ppt_text("20- and 60-day momentum\n126-to-20-day momentum\nReturn volatility\nAverage turnover", 9.10, 3.28, 3.48, 2.10, 20),
        _ppt_text("T1 · T2 · T3 · T5 · T6", 9.10, 5.57, 3.48, .28, 15, color=_ppt_muted),
        _ppt_text("Signs and usefulness can vary; no economic story guarantees alpha.", .65, 6.43, 12, .43, 20),
    ])
    notes.append("Explain profitability and cash generation, signed valuation and leverage, and price/trading mechanisms. F4 and T4 stay in the saved sixteen-column input axis but are unused. Negative equity and earnings are preserved. Annual factor choices and directions are published in the configuration; this reproduction runs no new search.")

    pages.append([
        *_ppt_header("Historical coverage defines the opportunity set", f"{span}  |  {int(summary['days']):,} sessions  |  {int(summary['stock_slots']):,} historical slots"),
        _ppt_text(f"16 saved columns  ·  14 finite candidates  ·  First signal: {_ppt_human_date(summary['first_signal'])}",
                  .65, 1.83, 12, .35, 18, True),
        _ppt_chart(charts, "coverage", .55, 2.38, 12.15, 3.55),
        _ppt_text("All methods share one historical pool; unavailable inputs stay missing.", .65, 6.18, 12, .45, 20),
        _ppt_text("Report period and both announcement dates are screened. Unrevised vendor vintages cannot be certified.", .65, 6.73, 12, .24, 12, color=_ppt_muted),
    ])
    notes.append("Describe the prepared market and financial inputs, historical membership, non-ST and listing-age guards, trading status, and the common fourteen-candidate finite mask. The full calendar starts before the first feasible signal. Financial timing screens all three date fields; it cannot independently exclude later supplier revisions. No future label availability enters the pool.")

    pipeline = []
    for x, text in zip([.55, 3.03, 5.51, 7.99, 10.47],
                       ["Prepared\ninputs", "Within-date\nranks", "Fixed annual\nfits", "Industry +\nlog size", "Causal EMA\nand rerank"]):
        pipeline += [_ppt_box(x, 1.96, 2.20, 1.04), _ppt_text(text, x + .12, 2.18, 1.96, .64, 19, True, _ppt_teal, "center")]
    model_rows = [["Method", "Score or fixed model", "EMA sessions"],
                  ["Fundamental", "Signed financial ranks", "20"],
                  ["Combined", "Equal financial / technical families", "40"],
                  ["Ridge", "alpha = 10,000", "20"],
                  ["LightGBM", "depth 3; 7 leaves; 56 trees", "40"]]
    pages.append([
        *_ppt_header("A fixed model and processing pipeline", "Predict raw 20-session returns; update the scores every market day"),
        *pipeline,
        _ppt_table(model_rows, .55, 3.38, [2.70, 6.55, 2.90], .49, 18, [3]),
        _ppt_text("Only mature earlier labels enter annual fits; parameters stay fixed.", .65, 6.35, 12, .45, 20),
    ])
    notes.append("Walk through ranks, annual fits, score residualization, and EMA. Ridge uses alpha 10000 and the tree uses 56 shallow boosting rounds. Training follows the absolute every-fifth-session grid and requires the t+21 exit strictly before prediction start. The 2018–2020 section is fitted replay, while annual forecasts from 2021 use mature prior labels. Scores and returns are not the same object: returns are never smoothed.")

    pages.append([
        *_ppt_header("Implementation and the matched benchmark", "Daily signals and valuation; continuous twenty-session target refreshes"),
        _ppt_text("Trade at the next open; keep the actual cash and blocked orders.", .65, 1.83, 12, .44, 20),
        _ppt_box(.55, 2.48, 3.90, 3.37), _ppt_box(4.72, 2.48, 3.90, 3.37), _ppt_box(8.89, 2.48, 3.90, 3.37),
        _ppt_text("PORTFOLIO & TIMING", .76, 2.71, 3.48, .34, 18, True, _ppt_teal),
        _ppt_text("Equal-weight Top 100\nNext-opening execution\nT+1 and price limits\nBlocked-order retries", .76, 3.24, 3.48, 2.13, 20),
        _ppt_text("MATCHED BENCHMARK", 4.93, 2.71, 3.48, .34, 18, True, _ppt_teal),
        _ppt_text("Same eligible stock pool\nSame twenty-day targets\nSame restrictions / costs\nCSI500 is auxiliary", 4.93, 3.24, 3.48, 2.13, 20),
        _ppt_text("TRADING FRICTIONS", 9.10, 2.71, 3.48, .34, 18, True, _ppt_teal),
        _ppt_text("Commission: 3 bp / side\nSlippage: 5 bp / side\nSell stamp: 10 → 5 bp\nChange: 28 Aug 2023", 9.10, 3.24, 3.48, 2.13, 20),
        _ppt_text("Twelve sourced corporate events use explicit claims and settlement dates.", .65, 6.18, 12, .45, 20),
        _ppt_text("Synthetic adjusted units approximate dividends; exact receipt taxes, board lots and market impact are omitted.", .65, 6.73, 12, .24, 12, color=_ppt_muted),
    ])
    notes.append("Explain the difference between a daily signal and a twenty-session target. Gross and net holdings and cash evolve separately. The equal-weight benchmark uses the same opportunity set, cadence, constraints and matched costs. Explain exchange and cash-exit claims, cash becoming investable only after payment, and the zero-recovery listing-end assumption for unresolved positions. Some cash dates use the documented fifth-session convention. CSI500 is a secondary reference.")

    pages.append(_ppt_two_charts(charts, "Gross and net returns over the complete history", span,
                                "gross_absolute", "net_absolute", "GROSS ABSOLUTE RETURN", "NET ABSOLUTE RETURN",
                                "Absolute wealth compounds daily returns; net books pay their own costs."))
    notes.append("Compare all four methods and the equal-weight benchmark in both full-history panels, including weaker methods. Absolute returns compound daily portfolio returns; the curves are not rebased at year boundaries. Explain the initial flat portion before feasible trading, the fitted early history, and the cost effect without describing the gross-minus-net wealth gap as a sum of fees.")

    pages.append(_ppt_two_charts(charts, "Gross and net excess over the same-cost benchmark", span,
                                "gross_excess", "net_excess", "GROSS CUMULATIVE DAILY EXCESS (pp)", "NET CUMULATIVE DAILY EXCESS (pp)",
                                "Excess sums daily return differences in percentage points."))
    notes.append("Show every method against its corresponding fee-matched benchmark. Excess is the arithmetic sum of daily return differences in percentage points. It is neither compounded relative wealth nor the difference between two compounded returns. A method can underperform in part of the history even if the final sum is positive. Do not omit unfavorable curves.")

    pages.append([
        *_ppt_header("Higher return still comes with drawdown risk", f"Net portfolios  |  {span}"),
        _ppt_text("PEAK-TO-TROUGH DRAWDOWN", .65, 1.88, 7.8, .30, 16, True),
        _ppt_text("RETURN AND VOLATILITY", 8.98, 1.88, 3.80, .30, 15, True),
        _ppt_chart(charts, "drawdown", .55, 2.28, 8.0, 3.7),
        _ppt_chart(charts, "risk_return", 8.98, 2.28, 3.8, 3.7),
        _ppt_text(f"Ridge worst drawdown: {_ppt_percent(full_dd, 1)}; benchmark: {_ppt_percent(benchmark_dd, 1)}.", .65, 6.35, 12, .45, 20),
    ])
    notes.append("Use the complete net-wealth drawdowns and return-risk scatter together. Compare all methods. Ridge's historical maximum drawdown is slightly deeper than the matched benchmark, so the higher return must not be presented as lower risk. Explain annualization, zero risk-free-rate Sharpe conventions, and why a single endpoint cannot summarize the path.")

    pages.append([
        *_ppt_header("Excess varies across months and calendar years", f"Monthly daily-excess sums and annual compounded net returns  |  {latest} through {end}"),
        _ppt_text("MONTHLY NET EXCESS · pp", .65, 1.88, 7.8, .30, 16, True),
        _ppt_text("CALENDAR-YEAR NET RETURN", 8.98, 1.88, 3.80, .30, 15, True),
        _ppt_chart(charts, "calendar_excess", .55, 2.28, 8.0, 3.65),
        _ppt_chart(charts, "annual_return", 8.98, 2.28, 3.8, 3.65),
        _ppt_text(f"Latest-year net excess: {_ppt_pp(latest_excess)}; individual months can lag.", .65, 6.35, 12, .45, 20),
        _ppt_text(f"{latest} is incomplete; future months are blank rather than zero.", 8.98, 5.98, 3.80, .38, 12, color=_ppt_muted),
    ])
    notes.append("Read the red and teal monthly cells and connect their sum to daily net excess. Gray or dashed future months are unobserved, not zero-return months. Annual returns compound, unlike the monthly excess sums. The latest year ends on 29 April and must not be compared to a full year as if lengths were identical.")

    pages.append(_ppt_two_charts(charts, "Rolling windows show changes in return and risk", "252-session windows | " + span,
                                "rolling_excess", "rolling_vol", "252-SESSION NET EXCESS (pp)", "252-SESSION ANNUALIZED NET VOLATILITY (%)",
                                "Relative performance and volatility change through time."))
    notes.append("Read the chart's stated rolling window rather than treating rolling observations as independent tests. Discuss weak relative-return intervals as well as strong ones, and compare the changing volatility. Warm-up windows are missing rather than filled. Rolling diagnostics do not create a new strategy or a new parameter choice.")

    pages.append(_ppt_two_charts(charts, "Turnover and trading costs remain material", span,
                                "turnover", "costs", "MONTHLY TURNOVER", "COST COMPONENTS · SUM OF DAILY OPENING-NAV RATIOS, pp",
                                "Costs alter cash, holdings and future returns in the net ledger."))
    notes.append("Define turnover as half of traded notional divided by opening wealth. Separate commission, slippage and stamp tax. Cost bars sum cost divided by each day's opening NAV, in percentage points; they are not a compounded wealth drag. Twenty-session targets do not eliminate retries, cash settlement or forced-event trading. Gross and net books can have different future holdings because costs constrain cash, so the compounded performance difference is not a pure fee sum. Daily doubled-slippage is a daily control, not a twenty-session stress.")

    pages.append([
        *_ppt_header("Signal diagnostics reveal overlap and instability", "Used-input dependence and cross-sectional prediction quality are different questions"),
        _ppt_text(f"{len(summary['selected_features'])} USED INPUTS · CORRELATION 2021–2025", .65, 1.84, 7.8, .30, 15, True),
        _ppt_text("ANNUAL RANK IC", 8.98, 1.84, 3.80, .30, 16, True),
        _ppt_chart(charts, "factor_corr", .55, 2.21, 8.0, 4.0),
        _ppt_chart(charts, "rank_ic", 8.98, 2.21, 3.8, 4.0),
        _ppt_text("Signal correlation and IC do not establish a tradable net return.", .65, 6.43, 12, .43, 20),
    ])
    notes.append("The matrix covers the union of twelve actual annual model inputs, averaging same-date Spearman correlations on the common pool every fifth session in 2021–2025. Annual model Rank IC uses only observable, mature twenty-session labels; an early fitted-replay IC is not independent evidence. Development diagnostic exits do not extend beyond 2025. Average ties are preserved. Explain why correlation, prediction IC, top-ranked cohort performance and actual fee-paying portfolio returns answer different questions.")

    pages.append([
        *_ppt_header("Do the highest scores identify stronger return cohorts?", "Mean 20-session same-pool excess (pp), before costs | overlapping cohorts"),
        _ppt_text("2021–2025", .65, 1.88, 5.8, .30, 17, True),
        _ppt_text(f"{latest} · MATURE LABELS ONLY", 6.90, 1.88, 5.9, .30, 17, True),
        _ppt_chart(charts, "quintile", .55, 2.28, 5.9, 3.6),
        _ppt_chart(charts, "quintile_2026", 6.90, 2.28, 5.9, 3.6),
        _ppt_text("Higher scores need not produce a monotone return pattern in every period.", .65, 6.18, 12, .45, 20),
        _ppt_text(f"Group 1 = low, group 5 = high. Last mature signal: {_ppt_human_date(summary['last_label_signal'])}.", .65, 6.74, 12, .23, 12, color=_ppt_muted),
    ])
    notes.append("Quintiles are formed from currently finite scores before looking at label availability; a fixed security slot breaks score ties. Within each original group, average only finite labels whose t+21 exit is observed. The 2021–2025 panel also requires exit no later than 2025-12-31, while 2026 requires exit no later than 2026-04-29. Report equal-date means of raw twenty-session return and same-pool label excess. Labels overlap, cohorts are not independent, and no trading costs are charged in this diagnostic. The 2026 panel has a shorter mature-label window, so retain coverage and date limitations.")

    latest_rows = [["Method", "Net %", "Excess pp"]]
    for method in _ppt_methods:
        latest_rows.append([_ppt_labels[method], f"{_ppt_metric(ctx, method, 'holdout') * 100:.2f}",
                            f"{_ppt_metric(ctx, method, 'holdout', 'cumulative_excess') * 100:+.2f}"])
    pages.append([
        *_ppt_header("Controls expose the effect of implementation choices", "Same opportunity set and stated costs; retain weaker comparisons"),
        _ppt_text("RAW / ADOPTED AND DAILY / TWENTY-SESSION CONTROLS", .65, 1.88, 7.8, .30, 14, True),
        _ppt_text(f"{latest} THROUGH {end.upper()}", 8.98, 1.88, 3.80, .30, 14, True),
        _ppt_chart(charts, "controls", .55, 2.28, 8.0, 3.70),
        _ppt_table(latest_rows, 8.98, 2.42, [1.70, .95, 1.15], .55, 16, [3]),
        _ppt_text("Excess is a daily sum versus the matching-cost benchmark.", 9.06, 5.84, 3.62, .50, 13, color=_ppt_muted),
        _ppt_text("Implementation choices affect both realised return and costs.", .65, 6.43, 12, .43, 20),
    ])
    notes.append("The upper control panel compares raw-signal and adopted twenty-session net returns for all four methods in 2021–2025. The lower control panel compares 2026 Ridge daily, every-twenty-session and daily doubled-slippage cumulative daily net excess. The doubled-slippage control remains daily and must not be called a twenty-session cost stress. The right-hand table keeps all four final methods and the matched benchmark visible. No new strategy is selected from these displays.")

    final = [*_ppt_header("Conclusions and the limits of the evidence", f"Actual historical data through {end}; no promise of future performance"),
             _ppt_box(.55, 2.00, 8.00 if "cap_weights" in charts else 12.15, 4.07)]
    final_width = 7.52 if "cap_weights" in charts else 11.67
    statements = ["Ridge beats the matched benchmark over 2021–2025.",
                  f"The latest-year net excess is {_ppt_pp(latest_excess)}.",
                  "Score residualization does not make holdings style neutral.",
                  "Vendor revisions and execution assumptions remain limits."]
    for number, statement in enumerate(statements):
        final.append(_ppt_text(f"{number + 1}.  {statement}", .79, 2.28 + number * .82, final_width, .72, 20))
    if "cap_weights" in charts:
        final += [_ppt_text("ACTUAL FINAL CAPITAL WEIGHTS", 8.98, 1.88, 3.80, .30, 14, True),
                  _ppt_chart(charts, "cap_weights", 8.98, 2.28, 3.8, 3.4)]
        if "positions" in ctx and ctx["positions"] is not None:
            positions = ctx["positions"]
            cap = positions[(positions["date"] == positions["date"].max())
                            & (positions["category"] == "cap")
                            & positions["bucket"].str.startswith("Q3")]
            ridge_q3 = float(cap.loc[cap["method"] == "ridge", "weight"].iloc[0])
            benchmark_q3 = float(cap.loc[cap["method"] == "benchmark", "weight"].iloc[0])
            final.append(_ppt_text(f"Q3: Ridge {ridge_q3:.1%}; benchmark {benchmark_q3:.1%}.",
                                   8.98, 5.94, 3.80, .30, 14, color=_ppt_muted))
    final.append(_ppt_text("Judge return, risk and implementation together.", .65, 6.43, 12, .43, 20))
    pages.append(final)
    notes.append("Conclude with both the historical return advantage and the modest latest-year excess. Neutralized scores need not produce market-cap-neutral held capital. The final capital chart uses actual net-NAV-normalized holdings, cash and claims, not idealized target weights; discuss the visible middle-cap overweight. Retain the supplier-revision caveat, synthetic-dividend approximation, corporate-event timing assumptions, board-lot/minimum-fee omissions and market impact. The early history is fitted replay and repeated development comparisons can overfit. Invite questions and let all group members explain their assigned part.")

    if len(pages) != 15 or sum(_ppt_seconds) != 1080:
        raise ValueError("The approved presentation is fifteen pages and eighteen minutes")
    presentation = _ppt_Presentation()
    presentation.slide_width = _ppt_Inches(_ppt_width)
    presentation.slide_height = _ppt_Inches(_ppt_height)
    presentation.core_properties.title = "Investment Group 9 | A-share cross-sectional strategy"
    presentation.core_properties.subject = "Fixed strategy, complete historical evidence and implementation limits"
    presentation.core_properties.author = "Group 9"
    base = _ppt_BytesIO()
    canvas = _ppt_Canvas(base, pagesize=(_ppt_width * 72, _ppt_height * 72))
    canvas.setTitle("Investment Group 9 | Presentation")
    canvas.setAuthor("Group 9")
    placements = []
    geometry = []
    text_boxes = tables = figures = 0
    for page_index, (elements, note, seconds) in enumerate(zip(pages, notes, _ppt_seconds)):
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = _ppt_RGBColor.from_string(_ppt_paper[1:])
        canvas.setFillColor(_ppt_colors.HexColor(_ppt_paper))
        canvas.rect(0, 0, _ppt_width * 72, _ppt_height * 72, fill=1, stroke=0)
        elements = [_ppt_text("GROUP 09  /  INVESTMENTS 1005", .55, .24, 11.5, .22, 10, True, _ppt_teal),
                    *elements,
                    _ppt_text("Matched costs and historical pool. Absolute returns compound; excess sums daily differences.", .55, 7.13, 11.7, .20, 10, color=_ppt_muted),
                    _ppt_text(f"{page_index + 1:02d}", 12.25, 7.11, .5, .24, 11, True, _ppt_teal)]
        for item in elements:
            if not (item["x"] >= 0 and item["y"] >= 0 and item["x"] + item["w"] <= _ppt_width + .001
                    and item["y"] + item["h"] <= _ppt_height + .001):
                raise ValueError(f"Slide element outside page {page_index + 1}: {item}")
            if item["type"] == "box":
                shape = slide.shapes.add_shape(_ppt_MSO_SHAPE.ROUNDED_RECTANGLE,
                                               _ppt_Inches(item["x"]), _ppt_Inches(item["y"]),
                                               _ppt_Inches(item["w"]), _ppt_Inches(item["h"]))
                shape.fill.solid()
                shape.fill.fore_color.rgb = _ppt_RGBColor.from_string(item["fill"][1:])
                shape.line.fill.background()
                shape.adjustments[0] = .12
                canvas.setFillColor(_ppt_colors.HexColor(item["fill"]))
                canvas.roundRect(item["x"] * 72, (_ppt_height - item["y"] - item["h"]) * 72,
                                 item["w"] * 72, item["h"] * 72, 8, fill=1, stroke=0)
            elif item["type"] == "chart":
                slide.shapes.add_picture(item["png"], _ppt_Inches(item["x"]), _ppt_Inches(item["y"]),
                                         width=_ppt_Inches(item["w"]), height=_ppt_Inches(item["h"]))
                placements.append((page_index, item))
                figures += 1
            elif item["type"] == "table":
                _ppt_draw_table(slide, canvas, item)
                tables += 1
            else:
                item["line_count"] = _ppt_draw_text(slide, canvas, item)
                text_boxes += 1
            geometry.append({"page": page_index + 1, **item})
        slide.notes_slide.notes_text_frame.text = f"Suggested speaking time: {seconds} seconds. {note}"
        canvas.showPage()
    canvas.save()
    pptx_path = project / "Presentation_Group9.pptx"
    pdf_path = project / "Presentation_Group9.pdf"
    presentation.save(pptx_path)
    with _ppt_fitz.open(stream=base.getvalue(), filetype="pdf") as document:
        for page_index, item in placements:
            with _ppt_fitz.open(item["pdf"]) as chart:
                box = _ppt_fitz.Rect(item["x"] * 72, item["y"] * 72,
                                     (item["x"] + item["w"]) * 72, (item["y"] + item["h"]) * 72)
                document[page_index].show_pdf_page(box, chart, 0)
        document.save(pdf_path, garbage=4, deflate=True)
    return {"pptx": pptx_path, "pdf": pdf_path, "slides": len(pages),
            "speaking_seconds": sum(_ppt_seconds), "slide_seconds": list(_ppt_seconds),
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
