"""Exploratory checks of the raw data: the evidence behind the cleaning rules and the leakage audit.

Run with: python -m src.explore
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.clean import arrival_date, load_raw  # noqa: E402


def show(title, result):
    print(f'\n## {title}\n{result}')


def main() -> None:
    hotel = load_raw()
    hotel['arrival_date'] = arrival_date(hotel)
    hotel['total_nights'] = hotel['stays_in_weekend_nights'] + hotel['stays_in_week_nights']
    hotel['total_guests'] = hotel['adults'] + hotel['children'].fillna(0) + hotel['babies']

    # ---------- Overview ----------
    show('Shape', hotel.shape)
    show('Missing values per column', hotel.isna().sum()[lambda x: x > 0])
    show('Fully duplicated rows (no booking ID exists)', hotel.duplicated().sum())
    show('Bookings per hotel', hotel['hotel'].value_counts())
    show('Cancellation rate: overall', hotel['is_canceled'].mean().round(2))
    show('Cancellation rate: per hotel', hotel.groupby('hotel')['is_canceled'].mean().round(2))
    show('reservation_status vs is_canceled (leakage)', pd.crosstab(hotel['reservation_status'], hotel['is_canceled']))

    # ---------- Dates and lead time ----------
    show('Arrival date range', (hotel['arrival_date'].min(), hotel['arrival_date'].max()))
    show('Days with no booking', (hotel['arrival_date'].max() - hotel['arrival_date'].min()).days + 1 - hotel['arrival_date'].nunique())
    show('Bookings per arrival day of week (0 = Monday)', hotel['arrival_date'].dt.dayofweek.value_counts().sort_index())
    show('Lead time (days)', hotel['lead_time'].describe().round(1))
    show('Rows with longest lead time',
         hotel.nlargest(5, 'lead_time')[['hotel', 'lead_time', 'arrival_date', 'market_segment', 'deposit_type', 'is_canceled']])
    lead_bins = [-1, 7, 30, 90, 180, 365, 800]
    show('Cancellation rate by lead time',
         hotel.groupby(pd.cut(hotel['lead_time'], lead_bins), observed=True)['is_canceled'].agg(['mean', 'size']).round(3))

    # ---------- Stay length and party size ----------
    zero_nights = hotel[hotel['total_nights'] == 0]
    show('Bookings with 0 nights: count / status / max ADR',
         (len(zero_nights), zero_nights['reservation_status'].value_counts().to_dict(), zero_nights['adr'].max()))
    show('Longest stay / stays over 30 nights', (hotel['total_nights'].max(), (hotel['total_nights'] > 30).sum()))
    for col in ['adults', 'children', 'babies']:
        show(f'Value counts of {col}', hotel[col].value_counts(dropna=False).sort_index().to_dict())
    show('Rows with missing children', hotel[hotel['children'].isna()][['hotel', 'arrival_date', 'adults', 'market_segment', 'is_canceled']])
    show('Bookings with 0 guests / 0 adults but some children',
         ((hotel['total_guests'] == 0).sum(), ((hotel['adults'] == 0) & (hotel['total_guests'] > 0)).sum()))
    # All Resort Hotel, all canceled, all $0 ADR: group block placeholders rather than real party sizes
    show('Rows with 10+ adults', hotel[hotel['adults'] >= 10][['hotel', 'adults', 'market_segment', 'adr', 'is_canceled']])
    show('Rows with 4+ children or 3+ babies',
         hotel[(hotel['children'] > 3) | (hotel['babies'] > 2)][['hotel', 'adults', 'children', 'babies', 'adr']])

    # ---------- Categorical variables ----------
    show('Meal (Undefined and SC both mean no meal package)', hotel['meal'].value_counts())
    show('Missing / unique countries', (hotel['country'].isna().sum(), hotel['country'].nunique()))
    show('Hotel of rows with missing country', hotel.loc[hotel['country'].isna(), 'hotel'].value_counts())
    # The source paper notes nationality is often only confirmed at check-in, so this partly reflects leakage
    show('Cancellation rate: Portugal (True) vs other', hotel.groupby(hotel['country'].eq('PRT'))['is_canceled'].mean().round(3))
    show('Market segment', hotel['market_segment'].value_counts())
    show('Distribution channel', hotel['distribution_channel'].value_counts())
    show('Cancellation rate by market segment',
         hotel.groupby('market_segment')['is_canceled'].agg(['mean', 'size']).round(2).sort_values('mean'))
    show('Cancellation rate by repeated guest', hotel.groupby('is_repeated_guest')['is_canceled'].mean().round(3))
    show('Share of 0 previous cancellations / bookings',
         (hotel[['previous_cancellations', 'previous_bookings_not_canceled']] == 0).mean().round(3).to_dict())
    # Only guests who arrive get a room assigned, so a different room is leakage
    diff_room = hotel['reserved_room_type'] != hotel['assigned_room_type']
    show('Cancellation rate: assigned room differs from reserved room', hotel.groupby(diff_room)['is_canceled'].mean().round(3))
    show('Cancellation rate by deposit type', hotel.groupby('deposit_type')['is_canceled'].agg(['mean', 'size']).round(4))

    non_refund = hotel[hotel['deposit_type'] == 'Non Refund']
    show('What is special about Non Refund bookings?', pd.Series({
        'City Hotel %': (non_refund['hotel'] == 'City Hotel').mean(),
        'Groups or Offline TA/TO %': non_refund['market_segment'].isin(['Groups', 'Offline TA/TO']).mean(),
        'Country PRT %': (non_refund['country'] == 'PRT').mean(),
        'Part of duplicated rows %': non_refund.duplicated(keep=False).mean(),
        'Median lead time': non_refund['lead_time'].median()}).round(3))
    show('Share missing / unique IDs of agent and company',
         (hotel[['agent', 'company']].isna().mean().round(3).to_dict(), hotel['agent'].nunique(), hotel['company'].nunique()))
    show('Cancellation rate: agent missing (True) vs not', hotel.groupby(hotel['agent'].isna())['is_canceled'].mean().round(3))

    # ---------- ADR (average daily rate) ----------
    show('ADR', hotel['adr'].describe().round(2))
    # One 5,400 ADR (next largest is 510) and one negative ADR
    show('Largest and negative ADR',
         pd.concat([hotel.nlargest(4, 'adr'), hotel[hotel['adr'] < 0]])[['hotel', 'adr', 'total_nights', 'market_segment']])
    show('Market segment of $0 ADR bookings', hotel.loc[hotel['adr'] == 0, 'market_segment'].value_counts().to_dict())

    # ---------- Timing and leakage evidence (used by the feature availability audit) ----------
    hotel['booking_date'] = hotel['arrival_date'] - pd.to_timedelta(hotel['lead_time'], unit='D')
    status_date = pd.to_datetime(hotel['reservation_status_date'])
    show('Reconstructed booking date range (arrival date - lead time)', (hotel['booking_date'].min(), hotel['booking_date'].max()))
    show('Days from arrival to reservation_status_date, by status',
         (status_date - hotel['arrival_date']).dt.days.groupby(hotel['reservation_status']).describe().round(1))
    show('Cancellation rate by parking spaces > 0 (recorded at arrival: perfect separation)',
         hotel.groupby(hotel['required_car_parking_spaces'] > 0)['is_canceled'].agg(['mean', 'size']).round(3))
    show('Cancellation rate by booking_changes > 0', hotel.groupby(hotel['booking_changes'] > 0)['is_canceled'].mean().round(3))
    show('Cancellation rate by special requests', hotel.groupby(hotel['total_of_special_requests'].clip(upper=3))['is_canceled'].mean().round(3))
    show('Cancellation rate by waiting list > 0', hotel.groupby(hotel['days_in_waiting_list'] > 0)['is_canceled'].mean().round(3))
    show('Cancellation rate when a field is missing (True) vs present',
         pd.DataFrame({c: hotel.groupby(hotel[c].isna())['is_canceled'].mean() for c in ['agent', 'company', 'country']}).round(3))

    # ---------- Raw distributions ----------
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    monthly = hotel.groupby(hotel['arrival_date'].dt.to_period('M')).size()
    axes[0, 0].plot(monthly.index.to_timestamp(), monthly, 'o-')
    axes[0, 0].set(xlabel='Arrival Month', ylabel='Number of Bookings', ylim=(0, monthly.max() * 1.2))
    axes[0, 0].tick_params(axis='x', rotation=30)
    axes[0, 1].hist(hotel['lead_time'], bins=100, log=True)
    axes[0, 1].set(xlabel='Lead Time (Days)', ylabel='Number of Bookings (log scale)')
    nights = hotel.loc[hotel['total_nights'] <= 30, 'total_nights'].value_counts()
    axes[1, 0].bar(nights.index, nights)
    axes[1, 0].set(xlabel='Total Nights (<= 30)', ylabel='Number of Bookings')
    axes[1, 1].hist(hotel['adr'], bins=100, log=True)
    axes[1, 1].set(xlabel='ADR', ylabel='Number of Bookings (log scale)')
    for ax in axes.flat:
        ax.grid(linestyle=':')
    fig.suptitle('Raw Data Distributions')
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(config.FIGURES_DIR / '00_raw_distributions.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
