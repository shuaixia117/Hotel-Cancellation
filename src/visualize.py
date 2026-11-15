"""Step 3: Describe the cleaned data with four figures."""
import pandas as pd
import matplotlib.pyplot as plt

clean = pd.read_csv('data/hotel_booking_clean.csv', parse_dates=['arrival_date'])


def save(ax, title, path):
    ax.grid(linestyle=':')
    ax.set_title(title)
    ax.figure.savefig(path, dpi=200, bbox_inches='tight')


# How many bookings per arrival date for each hotel?
daily = clean.groupby(['arrival_date', 'hotel']).size().unstack()
ax = daily.rolling(7).mean().plot(figsize=(10, 4), xlabel='Arrival Date', ylabel='Number of Bookings (7-Day MA)')
save(ax, 'Number of Hotel Bookings per Arrival Date', 'figures/fig1_bookings_per_date.png')

# How does median ADR change by arrival month?
monthly_adr = clean.groupby([clean['arrival_date'].dt.month, 'hotel'])['adr'].median().unstack()
ax = monthly_adr.plot(marker='o', figsize=(5, 4), xlabel='Arrival Month', ylabel='Median ADR', xticks=range(1, 13))
save(ax, 'Median ADR per Arrival Month', 'figures/fig2_adr_per_month.png')

# How does cancellation rate change with lead time?
lead_group = pd.cut(clean['lead_time'], bins=[-1, 7, 30, 90, 180, 365, 800],
                    labels=['0-7', '8-30', '31-90', '91-180', '181-365', '366+'])
cancel_rate = clean.groupby([lead_group, 'hotel'], observed=True)['is_canceled'].mean().unstack() * 100
ax = cancel_rate.plot.bar(figsize=(5, 4), rot=0, xlabel='Lead Time (Days)', ylabel='Cancellation Rate (%)')
ax.axhline(clean['is_canceled'].mean() * 100, linestyle='--', color='darkgray', label='Overall')
ax.legend()
save(ax, 'Cancellation Rate by Lead Time', 'figures/fig3_cancel_by_lead_time.png')

# What is the distribution of ADR after cleaning?
fig, ax = plt.subplots(figsize=(5, 4))
for h in ['City Hotel', 'Resort Hotel']:
    ax.hist(clean.loc[clean['hotel'] == h, 'adr'], bins=50, alpha=0.7, label=h)
ax.set(xlabel='ADR', ylabel='Number of Bookings')
ax.legend()
save(ax, 'ADR Distribution (Cleaned)', 'figures/fig4_adr_hist.png')
