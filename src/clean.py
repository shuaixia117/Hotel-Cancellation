"""Step 2: Clean the raw data and save data/hotel_booking_clean.csv."""
import pandas as pd

df = pd.read_csv('data/hotels_raw.csv')

# Drop rows that are invalid or placeholders (found in explore.py)
total_nights = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
total_guests = df['adults'] + df['children'] + df['babies']
is_dirty = (df['children'].isna() |
            df['market_segment'].eq('Undefined') |
            df['distribution_channel'].eq('Undefined') |
            (total_nights == 0) |
            (total_guests == 0) |
            (df['adults'] > 6) |
            (df['children'] > 3) |
            (df['babies'] > 2) |
            ~df['adr'].between(0, 1000) |
            (df['required_car_parking_spaces'] > 3))
print('Dropped rows:', is_dirty.sum(), f'({is_dirty.mean():.2%})')
df = df[~is_dirty].copy()

# Build new columns and fix types / labels
df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' +
                                    df['arrival_date_month'] + '-' +
                                    df['arrival_date_day_of_month'].astype(str), format='%Y-%B-%d')
df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['children'] = df['children'].astype(int)
df['meal'] = df['meal'].replace({'Undefined': 'SC'})               # both mean no meal package
df['country'] = df['country'].replace({'CN': 'CHN'}).fillna('UNK')
df[['agent', 'company']] = df[['agent', 'company']].astype('Int64')  # IDs; empty = none

# Drop the split date columns and arrival_date_week_number, keep the rest in a readable order
df = df[['hotel', 'is_canceled', 'lead_time', 'arrival_date',
         'stays_in_weekend_nights', 'stays_in_week_nights', 'total_nights',
         'adults', 'children', 'babies', 'meal', 'country',
         'market_segment', 'distribution_channel', 'is_repeated_guest',
         'previous_cancellations', 'previous_bookings_not_canceled',
         'reserved_room_type', 'assigned_room_type', 'booking_changes', 'deposit_type',
         'agent', 'company', 'days_in_waiting_list', 'customer_type', 'adr',
         'required_car_parking_spaces', 'total_of_special_requests',
         'reservation_status', 'reservation_status_date']]
df.to_csv('data/hotel_booking_clean.csv', index=False)
print('Saved data/hotel_booking_clean.csv', df.shape)
