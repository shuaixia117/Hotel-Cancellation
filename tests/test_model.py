import numpy as np
import pandas as pd
import pytest

from src import config, features, run_pipeline
from src import model as md
from tests.conftest import bookings


@pytest.fixture(scope='module')
def small_splits(small_data):
    return features.make_splits(small_data)


def test_model_selection_never_sees_test_rows(small_splits):
    """select_model only receives training and validation rows, and its result is reproducible."""
    first = run_pipeline.select_model(small_splits.train, small_splits.validation)
    used = set(first['train'].index) | set(first['validation'].index)
    assert not used & set(small_splits.test.index)
    assert (first['validation']['arrival_date'] < config.TEST_START).all()
    second = run_pipeline.select_model(small_splits.train, small_splits.validation)
    assert first['strict_terms'] == second['strict_terms'] and first['gbm_params'] == second['gbm_params']


def test_selection_rule_accepts_only_changes_that_improve_bic_and_validation_loss(small_splits):
    log = run_pipeline.select_model(small_splits.train, small_splits.validation)['selection_log']
    accepted = log[(log['step'] > 0) & log['accepted']]
    assert ((accepted['delta_bic'] < 0) & (accepted['delta_validation_log_loss'] < 0)).all()


def test_predictions_are_probabilities_and_baselines_use_training_rates():
    train = bookings(500, seed=0)
    train['is_canceled'] = (train['hotel'] == 'City Hotel').astype(int)  # City always cancels in training
    test = bookings(200, seed=1, is_canceled=0)
    enc = features.fit_encoder(train)
    tr, te = features.encode(train, enc), features.encode(test, enc)
    rates = md.hotel_rates(tr)
    p = md.predict_hotel_rate(rates, te, md.overall_rate(tr))
    assert set(np.unique(p)) <= {0.0, 1.0}   # training rates, not the test rate (which is 0)
    noisy = bookings(500, seed=3)
    m = md.fit_logit(['np.log1p(lead_time) * hotel', 'market_segment'], features.encode(noisy, features.fit_encoder(noisy)))
    q = md.predict(m, te)
    assert ((q > 0) & (q < 1)).all() and len(q) == len(te)


def test_scheme_c_uses_rule_for_non_refund_rows():
    df = bookings(300, seed=0)
    enc = features.fit_encoder(df)
    m = md.fit_logit(['np.log1p(lead_time)', 'hotel'], features.encode(df, enc))
    test = features.encode(bookings(10, seed=1, deposit_type=['Non Refund', 'No Deposit'] * 5), enc)
    p = md.predict_scheme_c(m, 0.99, test)
    assert np.allclose(p[::2], 0.99)
    assert np.allclose(p[1::2], md.predict(m, test.iloc[1::2]))


def test_drop_term_keeps_main_effects_of_an_interaction():
    terms = ['np.log1p(lead_time) * hotel', 'meal']
    assert md.drop_term(terms, 'np.log1p(lead_time) * hotel') == ['np.log1p(lead_time) + hotel', 'meal']
    assert md.drop_term(terms, 'meal') == ['np.log1p(lead_time) * hotel']


def test_hotel_effect_odds_ratio_matches_coefficients(small_splits):
    enc = features.fit_encoder(small_splits.refit)
    data = features.encode(small_splits.refit, enc)
    m = md.fit_logit(['np.log1p(lead_time) * hotel', 'market_segment'], data)
    effects = md.hotel_lead_effects(m, data.sample(200, random_state=0), np.array([0, 30]), n_draws=50)
    b = m.params
    expected = np.exp(b['hotel[T.Resort Hotel]'] + b['np.log1p(lead_time):hotel[T.Resort Hotel]'] * np.log1p([0, 30]))
    assert np.allclose(effects['odds_ratio'], expected)
    assert (effects['or_lower_95'] < effects['odds_ratio']).all() and (effects['odds_ratio'] < effects['or_upper_95']).all()
    assert ((effects['prob_city'] > 0) & (effects['prob_city'] < 1)).all()


def test_gbm_handles_unseen_categories(small_splits):
    enc = features.fit_encoder(small_splits.train)
    tr = features.encode(small_splits.train, enc)
    te = features.encode(small_splits.test.assign(market_segment='Brand new segment'), enc)
    p = md.predict_gbm(md.fit_gbm(tr, {'learning_rate': 0.1, 'max_leaf_nodes': 8}), te)
    assert ((p > 0) & (p < 1)).all()


def test_coefficient_table_has_odds_ratios():
    df = bookings(300, seed=0)
    m = md.fit_logit(['np.log1p(lead_time)'], features.encode(df, features.fit_encoder(df)))
    t = md.coefficient_table(m)
    assert np.allclose(t['odds_ratio'], np.exp(t['coef']))
    assert isinstance(t, pd.DataFrame) and list(t['term']) == ['Intercept', 'np.log1p(lead_time)']
