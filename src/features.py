"""Step 4: Prepare model features and the time-based train/test split.

Run this file to see the checks behind the modeling choices.
Excluded on purpose:
- Non Refund bookings (99.4% canceled, mostly group blocks).
- Leakage: reservation_status, reservation_status_date and assigned_room_type are recorded
  at or after the outcome. booking_changes and required_car_parking_spaces are only known after booking.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

SPLIT_DATE = '2017-01-01'  # learn from 2015-2016 arrivals, test on 2017 arrivals (as the hotel would use it)

# Reference level (first category) of each categorical variable
REFERENCE = {'hotel': 'City Hotel',
             'market_segment': 'Online TA',
             'customer_type': 'Transient',
             'agent_group': 'None',
             'meal': 'BB'}


def load_data():
    df = pd.read_csv('data/hotel_booking_clean.csv', parse_dates=['arrival_date'])
    df = df[df['deposit_type'] != 'Non Refund'].copy()

    df['month'] = df['arrival_date'].dt.month
    df['prev_cancel'] = (df['previous_cancellations'] > 0).astype(int)
    df['prev_stay'] = (df['previous_bookings_not_canceled'] > 0).astype(int)
    df['kids'] = ((df['children'] + df['babies']) > 0).astype(int)

    # agent has 333 IDs: keep the 10 largest agents in the training period, group the rest as 'Other'
    top_agents = df.loc[df['arrival_date'] < SPLIT_DATE, 'agent'].value_counts().head(10).index
    df['agent_group'] = 'Other'
    df.loc[df['agent'].isin(top_agents), 'agent_group'] = df['agent'].astype('Int64').astype(str)
    df.loc[df['agent'].isna(), 'agent_group'] = 'None'

    for col, level in REFERENCE.items():
        others = sorted(set(df[col]) - {level})
        df[col] = pd.Categorical(df[col], categories=[level] + others)
    return df


def split(df):
    return df[df['arrival_date'] < SPLIT_DATE], df[df['arrival_date'] >= SPLIT_DATE]


if __name__ == '__main__':
    df = load_data()
    train, test = split(df)
    print('All / train / test:', df.shape, train.shape, test.shape)
    print('Cancellation rate all / train / test:',
          df['is_canceled'].mean().round(3), train['is_canceled'].mean().round(3), test['is_canceled'].mean().round(3))
    print('\nAgent groups:\n', df['agent_group'].value_counts())

    # Which transformation of lead_time fits best?
    print('\nBIC of lead_time transformations:')
    for f in ['lead_time', 'np.sqrt(lead_time)', 'np.log1p(lead_time)']:
        print(' ', f, round(smf.logit('is_canceled ~ ' + f, data=train).fit(disp=0).bic, 1))

    # Is the log-odds of cancellation linear in log(1 + lead_time)?  Is the slope the same for both hotels?
    fig, ax = plt.subplots(figsize=(5, 4))
    for h in ['City Hotel', 'Resort Hotel']:
        sub = train[train['hotel'] == h]
        bins = pd.qcut(sub['lead_time'], 20, duplicates='drop')
        x = np.log1p(sub['lead_time']).groupby(bins, observed=True).mean()
        rate = sub['is_canceled'].groupby(bins, observed=True).mean()
        ax.plot(x, np.log(rate / (1 - rate)), 'o-', label=h)
    ax.set(xlabel='log(1 + Lead Time)', ylabel='Log-Odds of Cancellation', title='Empirical Log-Odds vs Lead Time')
    ax.legend()
    ax.grid(linestyle=':')
    fig.savefig('figures/p2_fig1_logodds_lead_time.png', dpi=200, bbox_inches='tight')

    # Are market_segment and distribution_channel redundant?
    print('\n', pd.crosstab(train['market_segment'], train['distribution_channel']))

    # Is the effect of a previous cancellation stable over time?
    quarter = df['arrival_date'].dt.to_period('Q').astype(str)
    rate = df.groupby([quarter, 'prev_cancel'])['is_canceled'].mean().unstack() * 100
    ax = rate.plot(marker='o', figsize=(5, 4), rot=45, xlabel='Arrival Quarter', ylabel='Cancellation Rate (%)')
    ax.axvline(5.5, linestyle='--', color='darkgray')  # between 2016Q4 and 2017Q1
    ax.legend(['No previous cancellation', 'Has previous cancellation', 'Train | Test'])
    ax.grid(linestyle=':')
    ax.set_title('Cancellation Rate by Previous Cancellation')
    ax.figure.savefig('figures/p2_fig2_prev_cancel_drift.png', dpi=200, bbox_inches='tight')
    print('\nCancellation rate (%) by quarter and previous cancellation:\n', rate.round(1))

    # Is the month effect consistent between 2015 and 2016?
    by_month = train.groupby([train['arrival_date'].dt.year, 'month'])['is_canceled'].mean().unstack(0) * 100
    print('\nCancellation rate (%) by month, 2015 vs 2016:\n', by_month.round(1))
