import numpy as np
import pandas as pd
import pytest

from src import clean, features


@pytest.fixture(scope='session')
def raw() -> pd.DataFrame:
    return clean.load_raw()


@pytest.fixture(scope='session')
def data() -> pd.DataFrame:
    """The cleaned data with derived features (built from the raw file, so tests do not depend on run order)."""
    return features.add_features(clean.clean_data(clean.load_raw()).assign(
        reservation_status_date=lambda d: pd.to_datetime(d['reservation_status_date'])))


@pytest.fixture(scope='session')
def small_data(data) -> pd.DataFrame:
    """A 25% sample that keeps model tests fast."""
    return data.sample(frac=0.25, random_state=0)


def bookings(n: int = 400, seed: int = 0, **overrides) -> pd.DataFrame:
    """Synthetic cleaned bookings with every column the feature code needs."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'hotel': rng.choice(['City Hotel', 'Resort Hotel'], n),
        'is_canceled': rng.integers(0, 2, n),
        'lead_time': rng.integers(0, 200, n),
        'arrival_date': pd.Timestamp('2016-01-01') + pd.to_timedelta(rng.integers(0, 600, n), unit='D'),
        'stays_in_weekend_nights': rng.integers(0, 3, n),
        'stays_in_week_nights': rng.integers(1, 5, n),
        'adults': rng.integers(1, 3, n), 'children': 0, 'babies': 0,
        'meal': rng.choice(['BB', 'HB', 'SC'], n), 'country': 'PRT',
        'market_segment': rng.choice(['Online TA', 'Direct', 'Groups'], n),
        'distribution_channel': rng.choice(['TA/TO', 'Direct'], n),
        'is_repeated_guest': 0, 'previous_cancellations': 0, 'previous_bookings_not_canceled': 0,
        'reserved_room_type': 'A', 'assigned_room_type': 'A', 'booking_changes': 0, 'deposit_type': 'No Deposit',
        'agent': rng.choice([9.0, 240.0, np.nan], n), 'company': np.nan, 'days_in_waiting_list': 0,
        'customer_type': 'Transient', 'adr': rng.uniform(50, 150, n), 'required_car_parking_spaces': 0,
        'total_of_special_requests': 0, 'reservation_status': 'Check-Out',
    })
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['reservation_status_date'] = df['arrival_date']
    for k, v in overrides.items():
        df[k] = v
    return features.add_features(df)
