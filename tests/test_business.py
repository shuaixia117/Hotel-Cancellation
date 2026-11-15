import numpy as np
import pytest

from src import business as bz

P = np.array([0.9, 0.6, 0.3, 0.1])
VALUE = np.array([400.0, 300.0, 200.0, 100.0])
Y = np.array([1, 0, 1, 0])


def scenario(**kw):
    params = {'name': 'test', 'cost_per_contact': 5.0, 'effectiveness': 0.2, 'margin_rate': 0.5, 'replacement_probability': 0.5}
    return bz.Scenario(**{**params, **kw})


def run(s, coverage=0.5, p=P, value=VALUE, y=Y):
    return bz.simulate_intervention(p, p, value, s, coverage, y)


def test_expected_profit_formula():
    s = scenario()
    expected = 0.9 * 0.2 * (1 - 0.5) * 0.5 * 400 - 5
    assert bz.expected_incremental_profit(np.array([0.9]), np.array([400.0]), s)[0] == pytest.approx(expected)


def test_targets_highest_risk_bookings():
    r = run(scenario(), coverage=0.5)
    assert r['targeted_bookings'] == 2
    assert r['expected_prevented_cancellations'] == pytest.approx((0.9 + 0.6) * 0.2)
    assert r['actual_cancellations_targeted'] == 1


def test_zero_cost():
    r = run(scenario(cost_per_contact=0))
    assert r['total_intervention_cost'] == 0
    assert r['expected_net_benefit'] == pytest.approx(r['expected_incremental_contribution']) and r['expected_net_benefit'] > 0


def test_zero_effectiveness():
    r = run(scenario(effectiveness=0))
    assert r['expected_incremental_contribution'] == 0 and r['realized_incremental_contribution'] == 0
    assert r['expected_net_benefit'] == -2 * 5.0


def test_zero_cancellation_probability():
    r = run(scenario(), p=np.zeros(4), y=np.zeros(4))
    assert r['expected_prevented_cancellations'] == 0 and r['expected_incremental_contribution'] == 0
    assert np.isnan(r['expected_break_even_effectiveness']) and np.isnan(r['realized_break_even_effectiveness'])


def test_zero_margin_and_full_replacement_give_no_contribution():
    for s in [scenario(margin_rate=0), scenario(replacement_probability=1)]:
        r = run(s)
        assert r['expected_incremental_contribution'] == 0
        assert r['expected_net_benefit'] == -r['total_intervention_cost']


def test_negative_profit_scenarios():
    assert run(scenario(cost_per_contact=1000))['expected_net_benefit'] < 0
    assert run(scenario(margin_rate=-0.2, cost_per_contact=0))['expected_net_benefit'] < 0


def test_empty_input():
    r = bz.simulate_intervention(np.array([]), np.array([]), np.array([]), scenario(), 0.5, np.array([]))
    assert r['targeted_bookings'] == 0 and r['total_intervention_cost'] == 0 and r['expected_net_benefit'] == 0
    assert np.isnan(r['expected_break_even_effectiveness'])


def test_break_even_effectiveness_gives_zero_net_benefit():
    r = run(scenario())
    at_break_even = run(scenario(effectiveness=r['expected_break_even_effectiveness']))
    assert at_break_even['expected_net_benefit'] == pytest.approx(0, abs=1e-9)


def test_coverage_bounds():
    assert run(scenario(), coverage=0)['targeted_bookings'] == 0
    assert run(scenario(), coverage=1)['targeted_bookings'] == 4
    with pytest.raises(ValueError):
        run(scenario(), coverage=1.5)


@pytest.mark.parametrize('bad', [{'effectiveness': 1.2}, {'effectiveness': -0.1}, {'replacement_probability': 2},
                                 {'cost_per_contact': -1}, {'margin_rate': 3}])
def test_invalid_parameters_are_rejected(bad):
    with pytest.raises(ValueError):
        scenario(**bad)


def test_invalid_probabilities_are_rejected():
    with pytest.raises(ValueError):
        bz.expected_incremental_profit(np.array([1.5]), np.array([100.0]), scenario())


def test_random_strategy_matches_its_expectation():
    curve = bz.coverage_curve({'model': P}, P, VALUE, Y, scenario(cost_per_contact=0), [1.0])
    model_all = curve.loc[curve['strategy'] == 'model', 'realized_net_benefit'].iloc[0]
    random_all = curve.loc[curve['strategy'] == 'Random (expected)', 'realized_net_benefit'].iloc[0]
    assert model_all == pytest.approx(random_all)   # targeting everyone: ranking does not matter


def test_risk_tier_cut_points_come_from_reference_predictions_only():
    ref = np.linspace(0, 1, 101)
    t1, cuts1 = bz.risk_tiers(ref, P, Y, VALUE)
    t2, cuts2 = bz.risk_tiers(ref, P, 1 - Y, VALUE)
    assert cuts1 == cuts2 == pytest.approx([0.5, 0.8])
    assert list(t1['bookings']) == [2, 1, 1]
    assert t1['share_of_cancellations'].sum() == pytest.approx(1)
