"""Scenario-based economics of contacting high-risk bookings.

Everything here is an ILLUSTRATIVE model. The data contains no retention experiments, so the
effect of an intervention is an assumption (effectiveness), not an estimate. ADR is revenue, not
profit; margin_rate converts room revenue into contribution margin, and replacement_probability
accounts for canceled rooms that would have been resold anyway.

Per targeted booking:
    expected incremental profit = P(cancel) x effectiveness x (1 - replacement_probability)
                                  x margin_rate x booking_value - cost_per_contact
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src import config
from src.evaluate import rank_order


@dataclass(frozen=True)
class Scenario:
    name: str
    cost_per_contact: float
    effectiveness: float
    margin_rate: float
    replacement_probability: float

    def __post_init__(self):
        if self.cost_per_contact < 0:
            raise ValueError('cost_per_contact must be >= 0')
        for field in ('effectiveness', 'replacement_probability'):
            if not 0 <= getattr(self, field) <= 1:
                raise ValueError(f'{field} must be between 0 and 1')
        if not -1 <= self.margin_rate <= 1:
            raise ValueError('margin_rate must be between -1 and 1')

    def value_per_prevented(self, booking_value) -> np.ndarray:
        """Contribution kept when one would-be cancellation is prevented."""
        return (1 - self.replacement_probability) * self.margin_rate * np.asarray(booking_value, dtype=float)


def scenarios() -> list[Scenario]:
    return [Scenario(name, **params) for name, params in config.SCENARIOS.items()]


def expected_incremental_profit(p_cancel, booking_value, scenario: Scenario) -> np.ndarray:
    p_cancel = np.asarray(p_cancel, dtype=float)
    if ((p_cancel < 0) | (p_cancel > 1)).any():
        raise ValueError('p_cancel must be between 0 and 1')
    return p_cancel * scenario.effectiveness * scenario.value_per_prevented(booking_value) - scenario.cost_per_contact


def simulate_intervention(score, p_cancel, booking_value, scenario: Scenario, coverage: float, y=None) -> dict:
    """Target the `coverage` share of bookings with the highest score.

    expected_*: uses the model's probabilities (what the hotel would plan with).
    realized_*: uses the actual outcomes of the evaluation bookings (how well the targeting really worked),
    still under the assumed effectiveness, margin and replacement.
    """
    if not 0 <= coverage <= 1:
        raise ValueError('coverage must be between 0 and 1')
    p_cancel, booking_value = np.asarray(p_cancel, dtype=float), np.asarray(booking_value, dtype=float)
    n = len(p_cancel)
    k = int(round(coverage * n))
    idx = rank_order(score)[:k] if n else np.array([], dtype=int)
    value = scenario.value_per_prevented(booking_value[idx])
    cost = k * scenario.cost_per_contact
    out = {'scenario': scenario.name, 'coverage': coverage, 'targeted_bookings': k,
           'expected_prevented_cancellations': float((p_cancel[idx] * scenario.effectiveness).sum()),
           'total_intervention_cost': cost,
           'expected_incremental_contribution': float((p_cancel[idx] * scenario.effectiveness * value).sum())}
    out['expected_net_benefit'] = out['expected_incremental_contribution'] - cost
    gross = (p_cancel[idx] * value).sum()
    out['expected_break_even_effectiveness'] = cost / gross if gross > 0 else np.nan
    if y is not None:
        y_t = np.asarray(y, dtype=float)[idx]
        realized_gross = (y_t * value).sum()
        out['actual_cancellations_targeted'] = int(y_t.sum())
        out['realized_prevented_cancellations'] = float(y_t.sum() * scenario.effectiveness)
        out['realized_incremental_contribution'] = float(realized_gross * scenario.effectiveness)
        out['realized_net_benefit'] = out['realized_incremental_contribution'] - cost
        out['realized_break_even_effectiveness'] = cost / realized_gross if realized_gross > 0 else np.nan
    return out


def coverage_curve(strategies: dict[str, np.ndarray], p_cancel, booking_value, y, scenario: Scenario,
                   coverages) -> pd.DataFrame:
    """Realized net benefit by coverage for several ranking strategies (same bookings, same assumptions)."""
    rows = []
    for name, score in strategies.items():
        for c in coverages:
            r = simulate_intervention(score, p_cancel, booking_value, scenario, c, y)
            rows.append({'strategy': name, 'scenario': scenario.name, 'coverage': c,
                         'targeted_bookings': r['targeted_bookings'], 'realized_net_benefit': r['realized_net_benefit']})
    # Random targeting: expected value of a random draw of the same size
    y, value = np.asarray(y, dtype=float), scenario.value_per_prevented(booking_value)
    for c in coverages:
        k = int(round(c * len(y)))
        rows.append({'strategy': 'Random (expected)', 'scenario': scenario.name, 'coverage': c, 'targeted_bookings': k,
                     'realized_net_benefit': k * ((y * value).mean() * scenario.effectiveness - scenario.cost_per_contact)
                     if len(y) else 0.0})
    return pd.DataFrame(rows)


def sensitivity(score, p_cancel, booking_value, y, base: Scenario, coverage: float,
                grid: dict[str, list[float]]) -> pd.DataFrame:
    """Change one assumption at a time around the base scenario."""
    rows = []
    for field, values in grid.items():
        for v in values:
            s = Scenario(**{**asdict(base), field: v, 'name': f'{field}={v}'})
            r = simulate_intervention(score, p_cancel, booking_value, s, coverage, y)
            rows.append({'parameter': field, 'value': v, 'coverage': coverage,
                         'expected_net_benefit': r['expected_net_benefit'], 'realized_net_benefit': r['realized_net_benefit']})
    return pd.DataFrame(rows)


def risk_tiers(p_reference, p_eval, y_eval, booking_value, quantiles=config.RISK_TIER_QUANTILES) -> tuple[pd.DataFrame, list[float]]:
    """Low / Medium / High tiers with cut points set on reference (training) predictions, not on test labels."""
    cuts = list(np.quantile(np.asarray(p_reference), quantiles))
    tier = pd.cut(np.asarray(p_eval), [-np.inf, *cuts, np.inf], labels=['Low', 'Medium', 'High'])
    df = pd.DataFrame({'tier': tier, 'p': p_eval, 'y': np.asarray(y_eval), 'value': np.asarray(booking_value)})
    table = df.groupby('tier', observed=False).agg(bookings=('y', 'size'), mean_predicted=('p', 'mean'),
                                                   actual_rate=('y', 'mean'), cancellations=('y', 'sum'),
                                                   mean_booking_value=('value', 'mean'), total_booking_value=('value', 'sum'))
    table['share_of_bookings'] = table['bookings'] / table['bookings'].sum()
    table['share_of_cancellations'] = table['cancellations'] / table['cancellations'].sum()
    table['canceled_booking_value'] = df[df['y'] == 1].groupby('tier', observed=False)['value'].sum()
    return table.reset_index(), cuts
