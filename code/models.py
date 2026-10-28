"""Train Group 9 models and construct their daily stock scores.

Only prepared inputs and the published strategy specification are required.
The initial 2018--2020 block is a fitted replay; forecasts from 2021 use
strictly mature historical labels. Portfolio execution is handled separately.
Lasso selects its penalty with purged, historical-only annual validation.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge, enet_path
from threadpoolctl import threadpool_limits


METHODS = ('fundamental', 'combined', 'ridge', 'lightgbm')


def _lasso_path(inputs, targets, settings):
    """A centered float64 Gram path, shared with the development protocol."""
    fractions = np.sort(np.asarray(settings['alpha_fractions'], dtype=np.float64))[::-1]
    with threadpool_limits(limits=1, user_api='blas'):
        x = np.asfortranarray(inputs, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        xmean, ymean = x.mean(axis=0), float(y.mean())
        x -= xmean
        y = np.ascontiguousarray(y - ymean)
        gram = np.ascontiguousarray(x.T @ x)
        xy = np.ascontiguousarray(x.T @ y)
        maximum = float(np.max(np.abs(xy)) / len(y))
        if maximum <= 0:
            raise ValueError('No feature-target covariance in Lasso training')
        alphas, coefficients, gaps, iterations = enet_path(
            x, y, l1_ratio=1., alphas=maximum * fractions, precompute=gram, Xy=xy,
            check_input=False, copy_X=False, selection='cyclic',
            max_iter=settings['max_iter'], tol=settings['tol'], return_n_iter=True)
        coefficients[:, fractions == 1.] = 0.
        if max(iterations) >= settings['max_iter']:
            raise RuntimeError('Lasso path did not converge')
        intercepts = ymean - xmean @ coefficients
    return {'fractions': fractions, 'alpha_max': maximum, 'alphas': alphas,
            'coefficients': coefficients, 'intercepts': intercepts,
            'dual_gaps': gaps, 'n_iter': iterations}


def _fit_lasso(inputs, targets, sampled, counts, days, cutoff, settings):
    """Purged previous-year alpha validation, with separately anchored windows."""
    row_days = np.repeat(sampled, counts)
    validation_start = (int(cutoff) // 10000 - 1) * 10000 + 101
    inner = days[row_days + 21] < validation_start
    validation = days[row_days] >= validation_start
    full = np.ones(len(row_days), dtype=bool)
    window = settings.get('training_window_years')
    if window is not None:
        inner &= days[row_days] >= (validation_start // 10000 - window) * 10000 + 101
        full &= days[row_days] >= (int(cutoff) // 10000 - window) * 10000 + 101
    if not inner.any() or not validation.any() or not full.any():
        raise ValueError('Lasso requires nonempty purged train, validation and refit sets')
    path = _lasso_path(inputs[inner], targets[inner], settings)
    valid_x = np.asarray(inputs[validation])
    valid_y = np.asarray(targets[validation], dtype=np.float64)
    valid_dates, valid_counts = np.unique(row_days[validation], return_counts=True)
    offsets = np.r_[0, np.cumsum(valid_counts)[:-1]]
    accum = np.zeros((len(path['fractions']), 3), dtype=np.float64)
    for begin, count in zip(offsets, valid_counts):
        x = np.asfortranarray(valid_x[begin:begin + count], dtype=np.float64)
        y = valid_y[begin:begin + count]
        with threadpool_limits(limits=1, user_api='blas'):
            predictions = x @ path['coefficients'] + path['intercepts']
        accum[:, 0] += np.mean(np.square(predictions - y[:, None]), axis=0)
        ranks = rankdata(predictions, method='average', axis=0).astype(np.float64)
        yrank = rankdata(y, method='average').astype(np.float64)
        ranks -= ranks.mean(axis=0)
        yrank -= yrank.mean()
        norm = np.sqrt(np.sum(ranks * ranks, axis=0) * np.sum(yrank * yrank))
        ic = np.zeros(len(path['fractions']))
        np.divide(ranks.T @ yrank, norm, out=ic, where=norm > 0)
        accum[:, 1] += ic
        accum[:, 2] += np.ptp(predictions.astype(np.float32), axis=0) > 0
    candidates = []
    for k, fraction in enumerate(path['fractions']):
        nonzero = int(np.count_nonzero(path['coefficients'][:, k]))
        candidates.append({'alpha_fraction': float(fraction), 'alpha': float(path['alphas'][k]),
                           'mean_daily_mse': float(accum[k, 0] / len(valid_dates)),
                           'mean_daily_rank_ic': float(accum[k, 1] / len(valid_dates)),
                           'nonzero_count': nonzero, 'validation_days': len(valid_dates),
                           'rankable_validation_days': int(accum[k, 2]),
                           'eligible_for_selection': nonzero > 0 and accum[k, 2] == len(valid_dates),
                           'dual_gap': float(path['dual_gaps'][k]), 'n_iter': int(path['n_iter'][k])})
    rankable = [k for k, row in enumerate(candidates) if row['eligible_for_selection']]
    if not rankable:
        raise ValueError('No Lasso candidate can rank stocks for the Top100 portfolio')
    metric = settings.get('selection_metric', 'mean_daily_mse')
    if metric in ('mean_rank_ic', 'mean_daily_rank_ic', 'rank_ic'):
        key = lambda k: (-candidates[k]['mean_daily_rank_ic'], -candidates[k]['alpha_fraction'])
    elif metric in ('mean_daily_mse', 'mse'):
        key = lambda k: (candidates[k]['mean_daily_mse'], -candidates[k]['alpha_fraction'])
    else:
        raise ValueError(f'Unknown Lasso selection metric: {metric}')
    chosen = min(rankable, key=key)
    selected = candidates[chosen]
    for k, row in enumerate(candidates):
        row['selected'] = k == chosen
    fitted = _lasso_path(inputs[full], targets[full], settings)
    coefficient = fitted['coefficients'][:, chosen]
    if not np.count_nonzero(coefficient):
        raise ValueError('Selected Lasso refit has no stock-ranking coefficients')
    model = {'coefficient': coefficient, 'intercept': float(fitted['intercepts'][chosen]),
             'alpha': float(fitted['alphas'][chosen])}
    full_days = np.unique(row_days[full])
    record = {'inner_train_first_signal': int(days[row_days[inner].min()]),
              'inner_train_latest_exit': int(days[row_days[inner].max() + 21]),
              'inner_train_rows': int(inner.sum()),
              'validation_start': int(days[valid_dates[0]]),
              'validation_latest_exit': int(days[valid_dates[-1] + 21]),
              'validation_rows': int(validation.sum()), 'validation_days': len(valid_dates),
              'alpha_fraction': selected['alpha_fraction'], 'alpha': model['alpha'],
              'alpha_max': fitted['alpha_max'], 'nonzero_count': int(np.count_nonzero(coefficient)),
              'null_signal': False, 'rankable_validation_days': selected['rankable_validation_days'],
              'unconstrained_mse_best_fraction': min(candidates, key=lambda row: row['mean_daily_mse'])['alpha_fraction'],
              'selection_metric': metric, 'training_window_years': window,
              'target_transform': settings.get('target_transform', 'raw'),
              'dual_gap': float(fitted['dual_gaps'][chosen]), 'n_iter': int(fitted['n_iter'][chosen]),
              'train_rows': int(full.sum()), 'train_sampled_first': int(days[full_days[0]]),
              'train_sampled_last': int(days[full_days[-1]]),
              'train_latest_exit': int(days[full_days[-1] + 21]), 'train_sampled_days': len(full_days),
              'grid_boundary': selected['alpha_fraction'] in
                  (min(candidates[k]['alpha_fraction'] for k in rankable),
                   max(candidates[k]['alpha_fraction'] for k in rankable))}
    return model, record, candidates


def train_lasso(data, config, output_dir):
    """Train only Lasso; existing models and their frozen scores are untouched."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    days = np.asarray(data['days'])
    eligible = np.array(data['base_eligible'], dtype=bool, copy=True)
    first = int(np.flatnonzero(eligible.sum(axis=1) >= config['models']['first_signal_min_stocks'])[0])
    eligible[:first] = False
    shape = eligible.shape
    raw = np.lib.format.open_memmap(output / 'raw_scores_lasso.npy', mode='w+',
                                    dtype=np.float32, shape=shape)
    raw[:] = np.nan
    last_year = int(days[-1]) // 10000
    label_stop = int(_mature_indices(days, last_year * 10000 + 101)[-1]) + 1
    labels = np.full(shape, np.nan, dtype=np.float32)
    labels[:label_stop] = data['forward_returns'][:label_stop]
    labels[~eligible] = np.nan
    for day in range(label_stop):
        finite = np.isfinite(labels[day])
        if finite.any():
            if config['method_specs']['lasso']['model_params'].get('target_transform', 'raw') == 'rank':
                labels[day, finite] = centered_rank(labels[day, finite])
            else:
                labels[day, finite] -= np.mean(labels[day, finite], dtype=np.float64)
    folds = [(2021, first, int(np.searchsorted(days, 20210101)), 20210101, True)]
    for year in range(2021, last_year + 1):
        start = int(np.searchsorted(days, year * 10000 + 101))
        stop = int(np.searchsorted(days, (year + 1) * 10000 + 101))
        folds.append((year, start, stop, int(days[start]), False))
    records, coefficients, validation_records = [], [], []
    rank_path = output / '_lasso_rank_inputs.npy'
    for year, start, stop, cutoff, replay in folds:
        selection = config['original_selections'][str(year)]
        lasso_features = config['method_specs']['lasso']['features']
        if set(lasso_features) != set(selection['features']):
            selection = {'features': lasso_features,
                         'indices': [config['feature_names'].index(name) for name in lasso_features]}
        ranked = _rank_panel(data, eligible, selection['indices'], stop, rank_path)
        mature = _mature_indices(days, cutoff)
        sampled = mature[mature % config['models']['training_stride'] == 0]
        valid = eligible[sampled] & np.isfinite(labels[sampled])
        nonempty = valid.any(axis=1)
        sampled, valid = sampled[nonempty], valid[nonempty]
        inputs = np.asarray(ranked[sampled][valid], dtype=np.float32)
        targets = labels[sampled][valid]
        model, record, candidates = _fit_lasso(
            inputs, targets, sampled, valid.sum(axis=1), days, cutoff,
            config['method_specs']['lasso']['model_params'])
        phase = 'in_sample_fitted_diagnostic' if replay else 'past_mature_labels_annual_refit'
        for day in range(start, stop):
            slots = np.flatnonzero(eligible[day])
            x = np.asfortranarray(ranked[day, slots], dtype=np.float64)
            with threadpool_limits(limits=1, user_api='blas'):
                prediction = (x @ model['coefficient'] + model['intercept']).astype(np.float32)
            if np.ptp(prediction) <= 0 or not np.isfinite(prediction).all():
                raise ValueError(f'Lasso cannot rank eligible stocks on {days[day]}')
            raw[day, slots] = prediction
        record.update(year=year, phase=phase, prediction_start=int(days[start]),
                      prediction_end=int(days[stop - 1]),
                      training_stride=config['models']['training_stride'],
                      feature_count=len(selection['features']))
        records.append(record)
        coefficients.extend({'year': year, 'phase': phase, 'feature': name,
                             'coefficient': float(value)}
                            for name, value in zip(selection['features'], model['coefficient']))
        validation_records.extend(dict(row, year=year, phase=phase) for row in candidates)
        print(f'Lasso {int(days[start])}--{int(days[stop-1])}: '
              f'alpha={model["alpha"]:.6g}, fraction={record["alpha_fraction"]:g}, '
              f'nonzero={record["nonzero_count"]}/{len(model["coefficient"])}', flush=True)
        del ranked, inputs, targets, model
        rank_path.unlink()
    raw.flush()
    processed = _processed_score(data, raw, eligible, config['method_specs']['lasso']['processing'])
    np.save(output / 'scores_lasso.npy', processed)
    for name, rows in [('lasso_fit_windows', records), ('lasso_coefficients', coefficients),
                       ('lasso_validation', validation_records)]:
        pd.DataFrame(rows).to_csv(output / f'{name}.csv', index=False)
    return {'scores': {'lasso': np.load(output / 'scores_lasso.npy', mmap_mode='r')},
            'raw_scores': {'lasso': raw}, 'eligible': eligible,
            'first_signal': first, 'first_signal_index': first, 'fit_windows': records}


def centered_rank(values):
    """Average-tie percentile ranks, centered at zero; missing stays missing."""
    result = np.full(values.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(values)
    count = int(finite.sum())
    if count:
        result[finite] = (rankdata(values[finite], method='average') - .5) / count - .5
    return result


def _mature_indices(days, cutoff):
    exits = np.arange(len(days)) + 21
    valid = exits < len(days)
    valid[valid] &= days[exits[valid]] < cutoff
    return np.flatnonzero(valid)


def _rank_panel(data, eligible, indices, stop, path):
    ranked = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                     shape=(stop, eligible.shape[1], len(indices)))
    ranked[:] = np.nan
    for day in range(stop):
        valid = eligible[day]
        for column, index in enumerate(indices):
            ranked[day, valid, column] = centered_rank(data['features'][day, valid, index])
    ranked.flush()
    return ranked


def _simple_scores(ranked, selection):
    signs = np.asarray([selection['directions'][name] for name in selection['features']], dtype=np.float32)
    fundamental_columns = np.asarray([name.startswith('F') for name in selection['features']])
    fundamental = np.mean(ranked[:, :, fundamental_columns] * signs[fundamental_columns],
                          axis=2, dtype=np.float32)
    if fundamental_columns.all():
        return fundamental, fundamental.copy()
    technical = np.mean(ranked[:, :, ~fundamental_columns] * signs[~fundamental_columns],
                        axis=2, dtype=np.float32)
    return fundamental, (fundamental + technical) * np.float32(.5)


def _predict(model, ranked, eligible, dates):
    prediction = np.full((len(dates), eligible.shape[1]), np.nan, dtype=np.float32)
    for begin in range(0, len(dates), 20):
        block_dates = dates[begin:begin + 20]
        valid = eligible[block_dates]
        if valid.any():
            block = prediction[begin:begin + len(block_dates)]
            inputs = ranked[block_dates][valid]
            if isinstance(model, lgb.LGBMRegressor):
                values = model.booster_.predict(inputs, num_threads=model.n_jobs)
            else:
                with threadpool_limits(limits=1, user_api='blas'):
                    values = model.predict(inputs)
            block[valid] = values.astype(np.float32)
    return prediction


def _rerank(values):
    for day in range(len(values)):
        values[day] = centered_rank(values[day])


def _smooth(scores, eligible, span):
    result = np.array(scores, dtype=np.float32, copy=True)
    result[~eligible | ~np.isfinite(result)] = np.nan
    if not span:
        return result
    _rerank(result)
    state = np.full(result.shape[1:], np.nan, dtype=np.float64)
    rate = 2.0 / (span + 1.0)
    for day in range(len(result)):
        current = np.array(result[day], dtype=np.float64, copy=True)
        valid = np.isfinite(current)
        continuing = valid & np.isfinite(state)
        current[continuing] = (current[continuing] - state[continuing]) * rate + state[continuing]
        state[valid] = current[valid]
        current[~valid] = np.nan
        result[day] = current
    _rerank(result)
    return result


def _processed_score(data, raw, eligible, processing):
    score = raw
    if processing['neutralize'] != 'none':
        score = np.full(raw.shape, np.nan, dtype=np.float32)
        for day in range(len(raw)):
            valid = eligible[day] & np.isfinite(raw[day])
            if not valid.any():
                continue
            industry = np.asarray(data['industry'][day, valid]).copy()
            industry[~np.isfinite(industry)] = -1
            _, groups = np.unique(industry, return_inverse=True)
            count = np.bincount(groups)
            values = centered_rank(raw[day, valid]).astype(np.float64)
            size = np.log(np.asarray(data['mcap'][day, valid], dtype=np.float64))
            values -= np.bincount(groups, weights=values)[groups] / count[groups]
            size -= np.bincount(groups, weights=size)[groups] / count[groups]
            norm = np.sum(size * size, dtype=np.float64)
            if norm > 0:
                values -= size * np.sum(size * values, dtype=np.float64) / norm
            score[day, valid] = centered_rank(values)
    return _smooth(score, eligible, processing['prediction_rank_ema_span'])


def train_models(data, config, output_dir):
    """Return five selected score panels and their unsmoothed baselines.

    The original four methods retain fixed annual selections and settings.
    Lasso repeats its specified historical validation, without forecast-year
    return selection. Neither path reuses cached model predictions.
    All generated score panels are written directly to output_dir.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    days = np.asarray(data['days'])
    eligible = np.array(data['base_eligible'], dtype=bool, copy=True)
    settings = config['models']
    first = int(np.flatnonzero(eligible.sum(axis=1) >= settings['first_signal_min_stocks'])[0])
    eligible[:first] = False
    shape = eligible.shape
    raw_scores = {method: np.lib.format.open_memmap(output / f'raw_scores_{method}.npy', mode='w+',
                                                   dtype=np.float32, shape=shape) for method in METHODS}
    for score in raw_scores.values():
        score[:] = np.nan
    fundamental_raw = np.lib.format.open_memmap(output / '_fundamental_raw.npy', mode='w+',
                                                dtype=np.float32, shape=shape)
    fundamental_raw[:] = np.nan
    last_year = int(days[-1]) // 10000
    latest_cutoff = int(days[np.searchsorted(days, last_year * 10000 + 101)])
    mature = _mature_indices(days, latest_cutoff)
    label_stop = int(mature[-1]) + 1
    labels = np.full(shape, np.nan, dtype=np.float32)
    labels[:label_stop] = data['forward_returns'][:label_stop]
    labels[~eligible] = np.nan
    for day in range(label_stop):
        valid = np.isfinite(labels[day])
        if valid.any():
            labels[day, valid] -= np.mean(labels[day, valid], dtype=np.float64)
    folds = [(2021, first, int(np.searchsorted(days, 20210101)), 20210101, True)]
    for year in range(2021, last_year + 1):
        start = int(np.searchsorted(days, year * 10000 + 101))
        stop = int(np.searchsorted(days, (year + 1) * 10000 + 101))
        folds.append((year, start, stop, int(days[start]), False))
    tree_spec = config['method_specs']['lightgbm']['model_params']
    ridge_alpha = config['method_specs']['ridge']['model_params']['alpha']
    records = []
    rank_path = output / '_rank_inputs.npy'
    for year, start, stop, cutoff, replay in folds:
        selection = config['original_selections'][str(year)]
        ranked = _rank_panel(data, eligible, selection['indices'], stop, rank_path)
        train = _mature_indices(days, cutoff)
        sampled = train[train % settings['training_stride'] == 0]
        valid = eligible[sampled] & np.isfinite(labels[sampled])
        nonempty = valid.any(axis=1)
        sampled, valid = sampled[nonempty], valid[nonempty]
        inputs = np.asarray(ranked[sampled][valid], dtype=np.float32)
        targets = labels[sampled][valid]
        with threadpool_limits(limits=1, user_api='blas'):
            ridge = Ridge(alpha=ridge_alpha, solver='cholesky', fit_intercept=True).fit(inputs, targets)
        tree = lgb.LGBMRegressor(
            objective='regression', n_estimators=tree_spec['boosting_rounds'],
            learning_rate=settings['learning_rate'], reg_lambda=1.0,
            n_jobs=settings['threads'], random_state=settings['seed'], verbosity=-1,
            max_bin=63, deterministic=True, force_col_wise=True,
            **{key: tree_spec[key] for key in ('max_depth', 'num_leaves', 'min_child_samples')},
        ).fit(inputs, targets)
        dates = np.arange(start, stop)
        fundamental, combined = _simple_scores(ranked, selection)
        raw_scores['fundamental'][start:stop] = fundamental[start:stop]
        raw_scores['combined'][start:stop] = combined[start:stop]
        raw_scores['ridge'][start:stop] = _predict(ridge, ranked, eligible, dates)
        raw_scores['lightgbm'][start:stop] = _predict(tree, ranked, eligible, dates)
        records.append({'phase': 'in_sample_fitted_diagnostic' if replay else 'past_mature_labels_annual_refit',
                        'prediction_start': int(days[start]), 'prediction_end': int(days[stop - 1]),
                        'train_sampled_first': int(days[sampled[0]]), 'train_sampled_last': int(days[sampled[-1]]),
                        'train_latest_exit': int(days[sampled[-1] + 21]),
                        'train_sampled_days': len(sampled), 'train_rows': len(targets),
                        'training_stride': settings['training_stride']})
        del ranked, inputs, targets, ridge, tree, fundamental, combined
        rank_path.unlink()

        selected = config['fold_selections']['fundamental'][str(year)]
        fund_shape = (stop - start, shape[1], len(selected['indices']))
        fund_inputs = np.full(fund_shape, np.nan, dtype=np.float32)
        for offset, day in enumerate(range(start, stop)):
            for column, index in enumerate(selected['indices']):
                fund_inputs[offset, eligible[day], column] = centered_rank(data['features'][day, eligible[day], index])
        selected_fundamental, _ = _simple_scores(fund_inputs, selected)
        fundamental_raw[start:stop] = selected_fundamental
        del fund_inputs, selected_fundamental
        print(f'Trained {int(days[start])}--{int(days[stop - 1])}: {records[-1]["train_rows"]:,} mature rows', flush=True)

    scores = {}
    for method in METHODS:
        source = fundamental_raw if method == 'fundamental' else raw_scores[method]
        processed = _processed_score(data, source, eligible, config['method_specs'][method]['processing'])
        np.save(output / f'scores_{method}.npy', processed)
        scores[method] = np.load(output / f'scores_{method}.npy', mmap_mode='r')
        raw_scores[method].flush()
    np.save(output / 'eligible.npy', eligible)
    pd.DataFrame(records).to_csv(output / 'model_fit_windows.csv', index=False)
    del fundamental_raw
    (output / '_fundamental_raw.npy').unlink()
    lasso = train_lasso(data, config, output)
    scores.update(lasso['scores'])
    raw_scores.update(lasso['raw_scores'])
    return {'scores': scores, 'raw_scores': raw_scores, 'eligible': eligible,
            'first_signal': first, 'first_signal_index': first, 'fit_windows': records}
