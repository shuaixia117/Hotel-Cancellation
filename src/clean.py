"""Clean the raw data and document what every cleaning rule removes."""
from pathlib import Path

import pandas as pd

from src import config

# name: (rule type, reason). Rule types:
#   definition error  - the value contradicts the field definition
#   placeholder       - looks like a system or group-block record rather than a real stay
#   scope             - a real booking type that the model is not meant to cover
#   subjective cap    - an extreme value removed by a chosen threshold
RULE_INFO = {
    'missing_children': ('definition error', 'children is missing (4 rows)'),
    'undefined_segment_or_channel': ('definition error', "market_segment or distribution_channel is 'Undefined'"),
    'zero_nights': ('scope', '0 nights with ADR 0: day use or complimentary, not an overnight booking'),
    'zero_guests': ('definition error', 'no adults, children or babies'),
    'adults_over_6': ('placeholder', '10-55 adults, all Resort Hotel, all canceled, ADR 0: group blocks'),
    'children_over_3': ('subjective cap', 'one booking with 10 children; likely a typo'),
    'babies_over_2': ('subjective cap', 'two bookings with 9 and 10 babies; likely typos'),
    'adr_outside_0_1000': ('definition error', 'negative ADR or 5,400 (next largest is 510)'),
    'parking_over_3': ('subjective cap', 'one booking with 8 parking spaces'),
}


# Columns of data/hotel_booking_clean.csv (arrival_date replaces the three split date columns)
CLEAN_COLUMNS = ['hotel', 'is_canceled', 'lead_time', 'arrival_date',
                 'stays_in_weekend_nights', 'stays_in_week_nights', 'total_nights',
                 'adults', 'children', 'babies', 'meal', 'country',
                 'market_segment', 'distribution_channel', 'is_repeated_guest',
                 'previous_cancellations', 'previous_bookings_not_canceled',
                 'reserved_room_type', 'assigned_room_type', 'booking_changes', 'deposit_type',
                 'agent', 'company', 'days_in_waiting_list', 'customer_type', 'adr',
                 'required_car_parking_spaces', 'total_of_special_requests',
                 'reservation_status', 'reservation_status_date']


def load_raw(path: Path = config.RAW_DATA) -> pd.DataFrame:
    return pd.read_csv(path)


def cleaning_flags(df: pd.DataFrame) -> pd.DataFrame:
    """One boolean column per cleaning rule: True means the row is removed by that rule."""
    nights = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    guests = df['adults'] + df['children'] + df['babies']
    return pd.DataFrame({
        'missing_children': df['children'].isna(),
        'undefined_segment_or_channel': df['market_segment'].eq('Undefined') | df['distribution_channel'].eq('Undefined'),
        'zero_nights': nights == 0,
        'zero_guests': guests == 0,
        'adults_over_6': df['adults'] > 6,
        'children_over_3': df['children'] > 3,
        'babies_over_2': df['babies'] > 2,
        'adr_outside_0_1000': ~df['adr'].between(0, 1000),
        'parking_over_3': df['required_car_parking_spaces'] > 3,
    })


def arrival_date(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df['arrival_date_year'].astype(str) + '-' + df['arrival_date_month'] + '-'
                          + df['arrival_date_day_of_month'].astype(str), format='%Y-%B-%d')


def booking_date(arrival: pd.Series, lead_time: pd.Series) -> pd.Series:
    """Booking date recovered from the field definitions (arrival date minus lead time in days).

    This is a reconstruction, not an original booking time stamp.
    """
    return arrival - pd.to_timedelta(lead_time, unit='D')


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[~cleaning_flags(raw).any(axis=1)].copy()
    df['arrival_date'] = arrival_date(df)
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['children'] = df['children'].astype(int)
    df['meal'] = df['meal'].replace({'Undefined': 'SC'})               # both mean no meal package
    df['country'] = df['country'].replace({'CN': 'CHN'}).fillna('UNK')
    df[['agent', 'company']] = df[['agent', 'company']].astype('Int64')  # IDs; empty = none
    return df[CLEAN_COLUMNS]


def cleaning_report(raw: pd.DataFrame) -> pd.DataFrame:
    """Rows removed by each rule, with the cancellation rate of the removed rows."""
    flags = cleaning_flags(raw)
    only_rule = flags.sum(axis=1) == 1
    rows = []
    for rule, (kind, reason) in RULE_INFO.items():
        hit = flags[rule]
        rows.append({'rule': rule, 'type': kind, 'rows_flagged': int(hit.sum()),
                     'rows_flagged_by_this_rule_only': int((hit & only_rule).sum()),
                     'cancel_rate_of_flagged': round(raw.loc[hit, 'is_canceled'].mean(), 3) if hit.any() else None,
                     'reason': reason})
    removed = flags.any(axis=1)
    rows.append({'rule': 'ANY RULE (total removed)', 'type': '', 'rows_flagged': int(removed.sum()),
                 'rows_flagged_by_this_rule_only': None,
                 'cancel_rate_of_flagged': round(raw.loc[removed, 'is_canceled'].mean(), 3), 'reason': ''})
    return pd.DataFrame(rows)


def distribution_shift(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Sample size, cancellation rate and main category shares before and after cleaning."""
    rows = [{'variable': 'rows', 'level': '', 'before': len(raw), 'after': len(clean)},
            {'variable': 'is_canceled', 'level': 'rate', 'before': raw['is_canceled'].mean(), 'after': clean['is_canceled'].mean()}]
    for col in ['hotel', 'deposit_type', 'market_segment', 'customer_type']:
        before, after = raw[col].value_counts(normalize=True), clean[col].value_counts(normalize=True)
        for level in before.index:
            rows.append({'variable': col, 'level': level, 'before': before[level], 'after': after.get(level, 0.0)})
    out = pd.DataFrame(rows)
    out[['before', 'after']] = out[['before', 'after']].astype(float).round(4)
    return out


def main() -> None:
    config.RESULTS_DIR.mkdir(exist_ok=True)
    raw = load_raw()
    clean = clean_data(raw)
    clean.to_csv(config.CLEAN_DATA, index=False)
    cleaning_report(raw).to_csv(config.RESULTS_DIR / 'data_cleaning_report.csv', index=False)
    distribution_shift(raw, clean).to_csv(config.RESULTS_DIR / 'data_cleaning_distribution.csv', index=False)
    print(f'Removed {len(raw) - len(clean)} of {len(raw)} rows; saved {config.CLEAN_DATA.name} {clean.shape}')


if __name__ == '__main__':
    main()
