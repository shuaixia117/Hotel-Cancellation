import numpy as np
import pandas as pd
import pytest

from src import evaluate as ev


def calibrated_sample(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, n)
    return (rng.random(n) < p).astype(int), p


def test_metrics_of_a_calibrated_model():
    y, p = calibrated_sample()
    m = ev.metrics(y, p)
    assert abs(m['calibration_intercept']) < 0.05 and abs(m['calibration_slope'] - 1) < 0.05
    assert abs(m['mean_predicted'] - m['actual_rate']) < 0.01
    assert 0.5 < m['roc_auc'] < 1 and 0 < m['brier'] < 0.25


def test_metrics_of_a_constant_prediction():
    y = np.array([0, 1, 1, 0, 1])
    m = ev.metrics(y, np.full(5, 0.6))
    assert m['roc_auc'] == 0.5 and np.isnan(m['calibration_slope'])
    assert m['calibration_intercept'] == pytest.approx(0.0, abs=1e-6)


def test_top_k_with_perfect_ranking():
    y = np.array([1] * 10 + [0] * 90)
    score = np.linspace(1, 0, 100)
    r = ev.top_k(y, score, 0.05)
    assert r['k'] == 5 and r['precision'] == 1 and r['recall'] == 0.5 and r['lift'] == pytest.approx(10)
    assert ev.top_k(y, score, 0.2)['recall'] == 1


def test_top_k_breaks_ties_reproducibly_and_rejects_empty_input():
    y = np.array([1, 0] * 50)
    a, b = ev.top_k(y, np.ones(100), 0.1), ev.top_k(y, np.ones(100), 0.1)
    assert a == b
    with pytest.raises(ValueError):
        ev.top_k(np.zeros(10), np.ones(10), 0.1)
    with pytest.raises(ValueError):
        ev.top_k(np.array([]), np.array([]), 0.1)


def test_top_k_table_includes_random_baseline():
    y, p = calibrated_sample(2000)
    t = ev.top_k_table(y, {'model': p}, [0.1, 0.2])
    rand = t[t['model'] == 'Random (expected)']
    assert np.allclose(rand['lift'], 1) and np.allclose(rand['recall'], [0.1, 0.2])
    assert (t.loc[t['model'] == 'model', 'lift'] > 1).all()


def test_capture_curve_runs_from_zero_to_one():
    y, p = calibrated_sample(1000)
    c = ev.capture_curve(y, p)
    assert c['share_captured'].iloc[0] == 0 and c['share_captured'].iloc[-1] == 1
    assert c['share_captured'].is_monotonic_increasing


def test_bootstrap_cis_contain_estimates_and_self_difference_is_zero():
    y, p = calibrated_sample(3000)
    groups = np.repeat(np.arange(300), 10)
    t = ev.bootstrap_metrics(y, {'model': p, 'noise': np.random.default_rng(1).random(3000)}, groups, 'model', n_boot=50)
    assert ((t['ci_lower_95'] <= t['estimate']) & (t['estimate'] <= t['ci_upper_95'])).all()
    own = t[t['model'] == 'model']
    assert np.allclose(own[['diff_vs_model', 'diff_ci_lower_95', 'diff_ci_upper_95']], 0)


def test_calibration_table_bins():
    y, p = calibrated_sample(1000)
    t = ev.calibration_table(y, p, n_bins=10)
    assert len(t) == 10 and t['n'].sum() == 1000 and t['mean_predicted'].is_monotonic_increasing


def test_daily_forecast_metrics():
    df = pd.DataFrame({'arrival_date': pd.to_datetime(['2017-01-01'] * 2 + ['2017-01-02'] * 2), 'is_canceled': [1, 0, 1, 1]})
    daily = ev.daily_totals(df, {'perfect': np.array([1, 0, 1, 1.0]), 'low': np.array([0, 0, 0.5, 0.5])})
    m = ev.daily_metrics(daily).set_index('forecast')
    assert m.loc['perfect', 'mae'] == 0 and m.loc['perfect', 'rmse'] == 0
    assert m.loc['low', 'bias'] == pytest.approx(-1.0)


def test_subgroup_table_flags_small_groups():
    df = pd.DataFrame({'is_canceled': [1, 0] * 100 + [1, 0, 0]})
    p = np.r_[np.full(200, 0.5), [0.9, 0.1, 0.2]]
    t = ev.subgroup_table(df, p, {'g': pd.Series(['big'] * 200 + ['small'] * 3)}).set_index('group')
    assert t.loc['small', 'unstable'] and not t.loc['big', 'unstable']
