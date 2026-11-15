import numpy as np
import pandas as pd

from src import config, features
from tests.conftest import bookings

T_VAL, T_TEST = config.VALIDATION_START, config.TEST_START


def test_splits_respect_boundaries(data):
    s = features.make_splits(data)
    assert (s.train['arrival_date'] < T_VAL).all()
    assert (s.validation['booking_date'] >= T_VAL).all() and (s.validation['arrival_date'] < T_TEST).all()
    assert (s.refit['arrival_date'] < T_TEST).all()
    assert (s.test['booking_date'] >= T_TEST).all()
    assert (s.test['arrival_date'] <= config.DATA_END).all()


def test_splits_do_not_overlap(data):
    s = features.make_splits(data)
    for a, b in [(s.train, s.validation), (s.train, s.test), (s.validation, s.test), (s.refit, s.test)]:
        assert not set(a.index) & set(b.index)


def test_in_flight_bookings_are_excluded_from_selection_and_test():
    """Booked before a cutoff but arriving after it: the outcome was unknown at the cutoff and the booking predates it."""
    df = bookings(4)
    df['arrival_date'] = pd.to_datetime(['2016-03-01', '2016-08-01', '2016-08-01', '2017-03-01'])
    df['lead_time'] = [10, 10, 100, 100]   # row 2 booked before T_VAL, arrives after; row 3 booked in 2016, arrives in 2017
    df = features.add_features(df)
    s = features.make_splits(df)
    assert list(s.train.index) == [0]
    assert list(s.validation.index) == [1]
    assert list(s.refit.index) == [0, 1, 2]
    assert list(s.test.index) == []
    assert list(s.legacy_test.index) == [3]


def test_encoder_learns_levels_and_agents_from_training_rows_only():
    train = bookings(600, seed=1)
    train['agent'] = np.random.default_rng(1).choice(np.arange(1, 16), len(train)).astype(float)  # 15 agents > TOP_AGENTS
    test = bookings(300, seed=2, agent=777.0, market_segment='Aviation')  # unseen agent and segment, frequent in test
    enc = features.fit_encoder(train)
    assert '777' not in enc.top_agents
    assert enc.top_agents == train['agent'].astype(int).astype(str).value_counts().head(config.TOP_AGENTS).index.tolist()
    out = features.encode(test, enc)
    assert set(out['agent_group']) == {'Other'}
    assert 'Aviation' not in set(out['market_segment'].astype(str))
    assert out[config.CATEGORICAL_COLUMNS].notna().all().all()


def test_unseen_level_maps_to_reference_when_no_other_level_exists():
    train = bookings(200, seed=3)
    enc = features.fit_encoder(train)
    assert 'Other' not in enc.levels['market_segment']
    out = features.encode(bookings(5, seed=4, market_segment='Undefined segment'), enc)
    assert set(out['market_segment'].astype(str)) == {'Online TA'}


def test_rare_training_levels_are_merged():
    train = bookings(300, seed=5)
    train.loc[train.index[:3], 'meal'] = 'FB'   # 3 rows < MIN_CATEGORY_COUNT
    enc = features.fit_encoder(train)
    assert 'FB' not in enc.levels['meal'] and 'Other' in enc.levels['meal']
    assert enc.levels['meal'][0] == 'BB'   # reference level first


def test_feature_sets_contain_no_outcome_information():
    forbidden = config.LEAKAGE_COLUMNS + config.EXCLUDED_COLUMNS
    terms = config.BASE_TERMS + config.CANDIDATE_TERMS + config.EXTENDED_TERMS + [config.LEGACY_FORMULA] + config.GBM_COLUMNS
    for term in terms:
        for col in forbidden:
            assert col not in term, (col, term)


def test_strict_set_excludes_extended_variables():
    strict = ' '.join(config.BASE_TERMS + config.CANDIDATE_TERMS) + ' ' + ' '.join(config.GBM_COLUMNS)
    for var in ['total_of_special_requests', 'booking_changes', 'days_in_waiting_list']:
        assert var not in strict


def test_row_features_do_not_depend_on_other_rows(data):
    subset = data.head(500)
    again = features.add_features(subset.drop(columns=['booking_date', 'month', 'prev_cancel', 'prev_stay', 'kids',
                                                       'weekend_share', 'has_company_id', 'is_non_refund', 'booking_value']))
    pd.testing.assert_frame_equal(again[subset.columns], subset)
