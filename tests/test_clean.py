import pandas as pd
import pytest

from src import clean


def test_booking_date_is_arrival_minus_lead_time():
    arrival = pd.Series(pd.to_datetime(['2017-01-10', '2016-03-01', '2016-03-01']))
    lead = pd.Series([9, 0, 1])
    result = clean.booking_date(arrival, lead)
    assert list(result) == list(pd.to_datetime(['2017-01-01', '2016-03-01', '2016-02-29']))


def test_reconstructed_booking_dates_are_plausible(data):
    assert (data['booking_date'] <= data['arrival_date']).all()
    assert data['booking_date'].min() >= pd.Timestamp('2013-06-01')   # max lead time is 737 days before 2015-07-01


def test_cancellations_are_resolved_by_arrival(data):
    """Basis of the temporal design: an outcome is known on the arrival date at the latest."""
    canceled = data['reservation_status'] != 'Check-Out'
    assert (data.loc[canceled, 'reservation_status_date'] <= data.loc[canceled, 'arrival_date']).all()


@pytest.mark.parametrize('rule, change', [
    ('missing_children', {'children': None}),
    ('undefined_segment_or_channel', {'market_segment': 'Undefined'}),
    ('zero_nights', {'stays_in_weekend_nights': 0, 'stays_in_week_nights': 0}),
    ('zero_guests', {'adults': 0, 'children': 0, 'babies': 0}),
    ('adults_over_6', {'adults': 7}),
    ('children_over_3', {'children': 4}),
    ('babies_over_2', {'babies': 3}),
    ('adr_outside_0_1000', {'adr': -1.0}),
    ('parking_over_3', {'required_car_parking_spaces': 4}),
])
def test_each_cleaning_rule_flags_its_row(raw, rule, change):
    valid = raw[~clean.cleaning_flags(raw).any(axis=1)].head(1)
    bad = valid.copy()
    for col, value in change.items():
        bad[col] = value
    flags = clean.cleaning_flags(pd.concat([valid, bad], ignore_index=True))
    assert not flags.iloc[0].any()
    assert flags.loc[1, rule]


def test_clean_data_removes_flagged_rows_and_fixes_labels(raw):
    cleaned = clean.clean_data(raw)
    assert len(cleaned) == len(raw) - clean.cleaning_flags(raw).any(axis=1).sum() == 118540
    assert list(cleaned.columns) == clean.CLEAN_COLUMNS
    assert cleaned['children'].dtype.kind == 'i'
    assert 'Undefined' not in set(cleaned['meal']) and 'CN' not in set(cleaned['country'])
    assert cleaned['country'].notna().all()


def test_cleaning_report_totals(raw):
    report = clean.cleaning_report(raw).set_index('rule')
    assert report.loc['ANY RULE (total removed)', 'rows_flagged'] == 850
    assert set(clean.RULE_INFO) <= set(report.index)
