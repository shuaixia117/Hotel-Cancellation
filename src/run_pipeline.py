"""Run the full analysis: clean -> split -> select (validation) -> refit -> test once -> business scenarios.

Usage (from any directory once the package is installed with `pip install -e .`):
    python -m src.run_pipeline
    python -m src.run_pipeline --n-boot 50     # fewer bootstrap replicates for a quick run
"""
import argparse
import json

import numpy as np
import pandas as pd

from src import business, clean, config, visualize
from src import evaluate as ev
from src import features as ft
from src import model as md

STRICT, EXTENDED, GBM = 'Logistic (strict)', 'Logistic (extended)', 'Gradient boosting'
OVERALL, HOTEL = 'Overall rate baseline', 'Hotel rate baseline'


def save(df: pd.DataFrame, name: str, **kwargs) -> None:
    df.to_csv(config.RESULTS_DIR / name, index=kwargs.pop('index', False), float_format='%.6g', **kwargs)


def check_splits(s: ft.Splits) -> None:
    """Fail loudly if the temporal design is violated."""
    assert s.train['arrival_date'].max() < config.VALIDATION_START
    assert s.validation['booking_date'].min() >= config.VALIDATION_START
    assert s.validation['arrival_date'].max() < config.TEST_START
    assert s.refit['arrival_date'].max() < config.TEST_START
    assert s.test['booking_date'].min() >= config.TEST_START
    assert not set(s.train.index) & set(s.validation.index)
    assert not set(s.refit.index) & set(s.test.index)


def non_refund_free(d: pd.DataFrame) -> pd.DataFrame:
    return d[d['is_non_refund'] == 0]


def scheme_rows(stage: str, y: np.ndarray, non_refund: np.ndarray, p_a: np.ndarray, p_c: np.ndarray) -> list[dict]:
    """Compare the Non Refund schemes on one evaluation window.

    A: one logistic model for all bookings with deposit_type as a feature.
    B: the logistic model only, applied to bookings that are not Non Refund (lower coverage).
    C: B plus a rule for Non Refund bookings (their training cancellation rate). On non Non Refund
       bookings B and C give identical predictions.
    """
    nr = np.asarray(non_refund) == 1
    rows = [{'stage': stage, 'scheme': 'A: one model with deposit_type', 'population': 'all bookings', **ev.metrics(y, p_a)},
            {'stage': stage, 'scheme': 'C: Non Refund rule + logistic', 'population': 'all bookings', **ev.metrics(y, p_c)},
            {'stage': stage, 'scheme': 'A: one model with deposit_type', 'population': 'excluding Non Refund',
             **ev.metrics(y[~nr], p_a[~nr])},
            {'stage': stage, 'scheme': 'B (= C): logistic without Non Refund', 'population': 'excluding Non Refund',
             **ev.metrics(y[~nr], p_c[~nr])}]
    if nr.any():
        rows += [{'stage': stage, 'scheme': 'A: one model with deposit_type', 'population': 'Non Refund only',
                  **ev.metrics(y[nr], p_a[nr], calibration=False)},
                 {'stage': stage, 'scheme': 'C: Non Refund rule', 'population': 'Non Refund only',
                  **ev.metrics(y[nr], p_c[nr], calibration=False)}]
    for r in rows:
        r['coverage_of_all_bookings'] = r['n'] / len(y)
    return rows


def select_model(train: pd.DataFrame, validation: pd.DataFrame) -> dict:
    """Every modeling choice: logistic terms and boosting settings. Only training and validation rows are passed in."""
    encoder = ft.fit_encoder(non_refund_free(train))
    train_s = ft.encode(non_refund_free(train), encoder)
    val_s = ft.encode(non_refund_free(validation), encoder)
    strict_terms, selection_log = md.select_terms(train_s, val_s, config.BASE_TERMS, config.CANDIDATE_TERMS)
    gbm_params, gbm_tuning = md.tune_gbm(train_s, val_s)
    return {'strict_terms': strict_terms, 'selection_log': selection_log, 'gbm_params': gbm_params,
            'gbm_tuning': gbm_tuning, 'encoder': encoder, 'train': train_s, 'validation': val_s}


def main(n_boot: int = config.N_BOOTSTRAP) -> None:
    config.RESULTS_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(exist_ok=True)

    # ---------- Data ----------
    clean.main()
    df = ft.load_clean()
    splits = ft.make_splits(df)
    check_splits(splits)
    save(ft.split_summary(splits), 'split_summary.csv')
    save(ft.unseen_level_report(splits), 'encoding_fallbacks.csv')
    print(ft.split_summary(splits).to_string(index=False))

    # ---------- Stage 1: model selection on train + validation (no test data) ----------
    sel = select_model(splits.train, splits.validation)
    strict_terms, train_s, val_s, enc_sel, gbm_params = sel['strict_terms'], sel['train'], sel['validation'], sel['encoder'], sel['gbm_params']
    extended_terms = strict_terms + config.EXTENDED_TERMS
    save(sel['selection_log'], 'model_selection.csv')
    save(sel['gbm_tuning'], 'gbm_tuning.csv')
    print('\nSelected strict terms:', strict_terms)

    # Non Refund schemes, compared on validation
    enc_all_sel = ft.fit_encoder(splits.train)
    train_all, val_all = ft.encode(splits.train, enc_all_sel), ft.encode(splits.validation, enc_all_sel)
    scheme_a = md.fit_logit(strict_terms + ['deposit_type'], train_all)
    strict_sel = md.fit_logit(strict_terms, train_s)
    nr_rate_sel = float(splits.train.loc[splits.train['is_non_refund'] == 1, 'is_canceled'].mean())
    scheme_table = scheme_rows('validation', val_all['is_canceled'].to_numpy(), val_all['is_non_refund'].to_numpy(),
                               md.predict(scheme_a, val_all),
                               md.predict_scheme_c(strict_sel, nr_rate_sel, ft.encode(splits.validation, enc_sel)))

    # Validation metrics of every candidate model (the comparison that is allowed to guide choices)
    val_preds = {OVERALL: np.full(len(val_s), md.overall_rate(train_s)),
                 HOTEL: md.predict_hotel_rate(md.hotel_rates(train_s), val_s, md.overall_rate(train_s)),
                 STRICT: md.predict(strict_sel, val_s),
                 EXTENDED: md.predict(md.fit_logit(extended_terms, train_s), val_s),
                 GBM: md.predict_gbm(md.fit_gbm(train_s, gbm_params), val_s)}
    comparison = [{'stage': 'validation', 'model': k, **ev.metrics(val_s['is_canceled'], p)} for k, p in val_preds.items()]

    # ---------- Stage 2: refit on every outcome known at TEST_START, evaluate the test set once ----------
    enc = ft.fit_encoder(non_refund_free(splits.refit))
    refit_s = ft.encode(non_refund_free(splits.refit), enc)
    test_s = ft.encode(non_refund_free(splits.test), enc)
    y = test_s['is_canceled'].to_numpy()

    strict = md.fit_logit(strict_terms, refit_s)
    extended = md.fit_logit(extended_terms, refit_s)
    gbm = md.fit_gbm(refit_s, gbm_params)
    base_rate, rates = md.overall_rate(refit_s), md.hotel_rates(refit_s)

    preds = {OVERALL: np.full(len(test_s), base_rate),
             HOTEL: md.predict_hotel_rate(rates, test_s, base_rate),
             STRICT: md.predict(strict, test_s),
             EXTENDED: md.predict(extended, test_s),
             GBM: md.predict_gbm(gbm, test_s)}
    comparison += [{'stage': 'test', 'model': k, **ev.metrics(y, p)} for k, p in preds.items()]
    comparison = pd.DataFrame(comparison)
    save(comparison, 'model_comparison.csv')
    print('\n', comparison[['stage', 'model', 'roc_auc', 'average_precision', 'log_loss', 'brier',
                            'mean_predicted', 'actual_rate']].round(4).to_string(index=False))

    ci = ev.bootstrap_metrics(y, preds, test_s['arrival_date'], reference=STRICT, n_boot=n_boot)
    save(ci, 'model_comparison_ci.csv')

    p = preds[STRICT]
    save(pd.concat([ev.calibration_table(y, preds[m]).assign(model=m) for m in [STRICT, GBM]]), 'calibration_table.csv')
    save(ev.top_k_table(y, {STRICT: p, GBM: preds[GBM], HOTEL: preds[HOTEL]}), 'topk_metrics.csv')
    curves = pd.concat([ev.capture_curve(y, s).assign(model=m) for m, s in
                        {STRICT: p, GBM: preds[GBM], HOTEL: preds[HOTEL]}.items()])
    save(curves, 'capture_curve.csv')

    lead_group = pd.cut(test_s['lead_time'], config.LEAD_BINS, labels=config.LEAD_LABELS)
    subgroups = ev.subgroup_table(test_s, p, {'hotel': test_s['hotel'],
                                              'arrival quarter': test_s['arrival_date'].dt.to_period('Q'),
                                              'lead time (days)': lead_group,
                                              'market segment': test_s['market_segment'],
                                              'customer type': test_s['customer_type']})
    save(subgroups, 'subgroup_performance.csv')

    daily = ev.daily_totals(test_s, {STRICT: p, HOTEL: preds[HOTEL]})
    save(daily, 'daily_forecast.csv', index=True)
    daily_table = ev.daily_metrics(daily)
    save(daily_table, 'daily_forecast_metrics.csv')

    coefs = md.coefficient_table(strict)
    save(coefs, 'model_coefficients.csv')
    reference = refit_s.sample(3000, random_state=config.SEED)
    interaction = md.hotel_lead_effects(strict, reference, np.array([0, 1, 3, 7, 14, 30, 60, 90, 120, 180, 240, 300, 365]))
    save(interaction, 'hotel_lead_interaction.csv')

    # Non Refund schemes on the test set (the scheme was chosen on validation)
    enc_all = ft.fit_encoder(splits.refit)
    refit_all, test_all = ft.encode(splits.refit, enc_all), ft.encode(splits.test, enc_all)
    scheme_a_final = md.fit_logit(strict_terms + ['deposit_type'], refit_all)
    nr_rate = float(splits.refit.loc[splits.refit['is_non_refund'] == 1, 'is_canceled'].mean())
    scheme_table += scheme_rows('test', test_all['is_canceled'].to_numpy(), test_all['is_non_refund'].to_numpy(),
                                md.predict(scheme_a_final, test_all),
                                md.predict_scheme_c(strict, nr_rate, ft.encode(splits.test, enc)))
    save(pd.DataFrame(scheme_table), 'nonrefund_schemes.csv')

    # Robustness: exact duplicate rows removed from the training window
    dedup = refit_s.loc[non_refund_free(splits.refit).drop_duplicates(subset=clean.CLEAN_COLUMNS).index]
    strict_dedup = md.fit_logit(strict_terms, dedup)
    dedup_table = coefs[['term', 'coef']].merge(md.coefficient_table(strict_dedup)[['term', 'coef']],
                                               on='term', suffixes=('_all_rows', '_deduplicated'))
    # Pre-declared rule: a coefficient is robust if it keeps its sign and moves by at most 0.2 on the log-odds scale
    change = dedup_table['coef_deduplicated'] - dedup_table['coef_all_rows']
    dedup_table['robust_to_dedup'] = (np.sign(dedup_table['coef_deduplicated']) == np.sign(dedup_table['coef_all_rows'])) \
        & (change.abs() <= 0.2)
    dedup_metrics = pd.DataFrame([{'fit': 'all rows', 'train_rows': len(refit_s), **ev.metrics(y, p, False)},
                                  {'fit': 'deduplicated', 'train_rows': len(dedup),
                                   **ev.metrics(y, md.predict(strict_dedup, test_s), False)}])
    save(dedup_table, 'dedup_coefficients.csv')
    save(dedup_metrics, 'dedup_metrics.csv')


    # ---------- Business scenarios (illustrative assumptions) ----------
    value = test_s['booking_value'].to_numpy()
    tiers, cuts = business.risk_tiers(md.predict(strict, refit_s), p, y, value)
    save(tiers, 'risk_tiers.csv')
    scen_rows = [business.simulate_intervention(p, p, value, s, c, y) for s in business.scenarios() for c in config.COVERAGE_GRID]
    scen_table = pd.DataFrame(scen_rows)
    save(scen_table, 'business_scenarios.csv')
    # Predefined strategies, compared on the same test bookings (comparative scenario analysis, no policy search)
    strategies = {'Logistic: risk': p, 'Logistic: expected value (risk x booking value)': p * value, 'Hotel rate baseline': preds[HOTEL]}
    coverage_grid = np.round(np.arange(0.01, 1.001, 0.01), 2)
    curve = pd.concat([business.coverage_curve(strategies, p, value, y, s, coverage_grid) for s in business.scenarios()])
    save(curve, 'business_coverage_curve.csv')
    base = next(s for s in business.scenarios() if s.name == 'Base')
    sens = business.sensitivity(p, p, value, y, base, 0.2, {
        'effectiveness': [0.05, 0.1, 0.2, 0.3, 0.4], 'replacement_probability': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'margin_rate': [0.1, 0.2, 0.4, 0.6], 'cost_per_contact': [1.0, 3.0, 5.0, 10.0, 20.0]})
    save(sens, 'business_sensitivity.csv')

    # Decision policy fixed in advance: contact a booking if its expected incremental profit is positive.
    # Checked on validation (selection-stage model), then applied unchanged to the test set.
    policy = pd.DataFrame(
        [{'stage': 'validation', **business.threshold_policy(val_preds[STRICT], val_s['booking_value'], sc, val_s['is_canceled'])}
         for sc in business.scenarios()]
        + [{'stage': 'test', **business.threshold_policy(p, value, sc, y)} for sc in business.scenarios()])
    save(policy, 'business_policy.csv')

    # ---------- Headline numbers ----------
    def ci_of(model, metric):
        r = ci[(ci['model'] == model) & (ci['metric'] == metric)].iloc[0]
        return {'estimate': r['estimate'], 'ci_95': [r['ci_lower_95'], r['ci_upper_95']]}

    test_rows = comparison[comparison['stage'] == 'test'].set_index('model')
    headline = {
        'selected_strict_terms': strict_terms,
        'gbm_params': gbm_params,
        'test_set': {'rows': len(test_s), 'cancel_rate': float(y.mean()),
                     'definition': 'bookings made from 2017-01-01 (arrivals up to 2017-08-31), Non Refund excluded'},
        'strict_logistic': {m: ci_of(STRICT, m) for m in ['roc_auc', 'average_precision', 'log_loss', 'brier']},
        'calibration': test_rows.loc[STRICT, ['mean_predicted', 'actual_rate', 'calibration_intercept', 'calibration_slope']].to_dict(),
        'test_metrics_by_model': test_rows[['roc_auc', 'average_precision', 'log_loss', 'brier']].to_dict(orient='index'),
        'risk_tier_cut_points': cuts,
        'business_base_top20': {k: v for k, v in scen_table[(scen_table['scenario'] == 'Base') & (scen_table['coverage'] == 0.2)]
                                .iloc[0].items() if k != 'scenario'},
        'business_threshold_policy': policy.to_dict(orient='records'),
    }
    with open(config.RESULTS_DIR / 'test_metrics.json', 'w') as f:
        json.dump(headline, f, indent=2, default=float)

    # ---------- Figures ----------
    visualize.descriptive(df, train_s, non_refund_free(df))
    visualize.temporal_design(df)
    visualize.model_comparison(ci)
    visualize.roc_pr(y, {k: preds[k] for k in [STRICT, EXTENDED, GBM, HOTEL]})
    visualize.calibration(y, {STRICT: p, GBM: preds[GBM]})
    visualize.capture(curves)
    visualize.interaction(interaction, coefs)
    visualize.odds_ratios(coefs)
    visualize.subgroups(subgroups)
    visualize.daily(daily)
    visualize.net_benefit(curve, policy)
    print('\nDone. Results in', config.RESULTS_DIR, 'and figures in', config.FIGURES_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-boot', type=int, default=config.N_BOOTSTRAP, help='bootstrap replicates for test CIs')
    main(parser.parse_args().n_boot)
