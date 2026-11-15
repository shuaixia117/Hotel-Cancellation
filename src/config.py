"""Central configuration: paths, split dates, feature sets, evaluation and business settings.

Change settings here so every module uses the same values.
"""
from pathlib import Path

import pandas as pd

# ---------- Paths (independent of the working directory) ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
FIGURES_DIR = ROOT / 'figures'
RESULTS_DIR = ROOT / 'results'
RAW_DATA = DATA_DIR / 'hotels_raw.csv'
CLEAN_DATA = DATA_DIR / 'hotel_booking_clean.csv'

SEED = 42

# ---------- Temporal design (see docs/methodology.md) ----------
# The prediction is made on the booking date. A model used from date T can only learn from
# bookings whose outcome is known by T, i.e. bookings that arrived before T (every cancellation
# in the data happens on or before the arrival date).
DATA_START = pd.Timestamp('2015-07-01')      # first arrival date in the data
DATA_END = pd.Timestamp('2017-08-31')        # last arrival date in the data
VALIDATION_START = pd.Timestamp('2016-07-01')  # model selection: train on arrivals before, validate on bookings after
TEST_START = pd.Timestamp('2017-01-01')        # final model: train on arrivals before, test on bookings after

# ---------- Categorical encoding (learned on the training window only) ----------
TOP_AGENTS = 10          # agents with their own level; the rest become 'Other'
MIN_CATEGORY_COUNT = 30  # training levels rarer than this are merged into 'Other'
REFERENCE_LEVELS = {'hotel': 'City Hotel',
                    'market_segment': 'Online TA',
                    'distribution_channel': 'TA/TO',
                    'customer_type': 'Transient',
                    'agent_group': 'None',
                    'meal': 'BB',
                    'reserved_room_type': 'A',
                    'deposit_type': 'No Deposit'}
CATEGORICAL_COLUMNS = list(REFERENCE_LEVELS)

# ---------- Feature sets (see the feature availability audit in docs/methodology.md) ----------
# Outcome or post-outcome information: never used as a feature.
LEAKAGE_COLUMNS = ['is_canceled', 'reservation_status', 'reservation_status_date',
                   'assigned_room_type', 'required_car_parking_spaces']
# Excluded from every model because availability at booking time is doubtful (see audit).
EXCLUDED_COLUMNS = ['country']

# Strict booking-time model: starting terms, always included before selection.
BASE_TERMS = ['np.log1p(lead_time) * hotel', 'market_segment', 'customer_type',
              'agent_group', 'prev_stay', 'kids', 'meal']
# Strict booking-time candidates, tested one at a time on the validation window.
CANDIDATE_TERMS = ['I(np.log1p(lead_time) ** 2)', 'prev_cancel', 'is_repeated_guest', 'C(month)',
                   'distribution_channel', 'reserved_room_type', 'np.log1p(adr)', 'adults',
                   'np.log1p(total_nights)', 'weekend_share', 'has_company_id']
# Extended set: fields that can change after booking. Higher leakage risk, reported as sensitivity only.
EXTENDED_TERMS = ['total_of_special_requests', 'np.log1p(booking_changes)', 'np.log1p(days_in_waiting_list)']

# Columns given to the gradient boosting benchmark: the same booking-time information
# that the strict logistic regression could choose from.
GBM_COLUMNS = ['hotel', 'lead_time', 'month', 'stays_in_weekend_nights', 'stays_in_week_nights',
               'adults', 'children', 'babies', 'meal', 'market_segment', 'distribution_channel',
               'customer_type', 'agent_group', 'reserved_room_type', 'is_repeated_guest',
               'previous_cancellations', 'previous_bookings_not_canceled', 'adr', 'has_company_id']
GBM_GRID = [{'learning_rate': lr, 'max_leaf_nodes': leaves} for lr in (0.02, 0.05, 0.1) for leaves in (2, 4, 8, 15, 31)]

# ---------- Evaluation ----------
N_BOOTSTRAP = 500                    # day-clustered bootstrap replicates on the test set
TOP_K = [0.05, 0.10, 0.20, 0.30]     # share of bookings flagged as highest risk
RISK_TIER_QUANTILES = (0.5, 0.8)     # Low: bottom 50%, Medium: next 30%, High: top 20% of training risk
LEAD_BINS = [-1, 7, 30, 90, 180, 800]
LEAD_LABELS = ['0-7', '8-30', '31-90', '91-180', '181+']
MIN_SUBGROUP_EVENTS = 50             # subgroups with fewer cancellations are flagged as unstable

# ---------- Business scenarios (ILLUSTRATIVE ASSUMPTIONS, not hotel data) ----------
# cost_per_contact: cost of one retention action, in the same currency units as ADR
# effectiveness: share of would-be cancellations that the action prevents
# margin_rate: incremental contribution margin as a share of room revenue (ADR x nights)
# replacement_probability: chance that a canceled room would have been resold anyway
SCENARIOS = {
    'Conservative': {'cost_per_contact': 10.0, 'effectiveness': 0.10, 'margin_rate': 0.30, 'replacement_probability': 0.60},
    'Base': {'cost_per_contact': 5.0, 'effectiveness': 0.20, 'margin_rate': 0.40, 'replacement_probability': 0.40},
    'Optimistic': {'cost_per_contact': 3.0, 'effectiveness': 0.30, 'margin_rate': 0.50, 'replacement_probability': 0.20},
}
COVERAGE_GRID = [0.05, 0.10, 0.20, 0.30, 0.50]
