"""All figures. Each function writes one PNG to figures/."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from src import config  # noqa: E402
from src.evaluate import calibration_table  # noqa: E402

HOTEL_COLORS = {'City Hotel': '#1f77b4', 'Resort Hotel': '#ff7f0e'}


def _save(fig, name: str, title: str | None = None) -> None:
    for ax in fig.axes:
        ax.grid(linestyle=':', alpha=0.7)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------- Descriptive figures ----------
def descriptive(df: pd.DataFrame, train: pd.DataFrame, refundable: pd.DataFrame) -> None:
    # Bookings per arrival date
    fig, ax = plt.subplots(figsize=(10, 4))
    daily = df.groupby(['arrival_date', 'hotel']).size().unstack()
    daily.rolling(7).mean().plot(ax=ax, color=HOTEL_COLORS)
    ax.set(xlabel='Arrival Date', ylabel='Bookings per Day (7-Day Moving Average)')
    _save(fig, '02_bookings_per_arrival_date.png', 'Number of Bookings per Arrival Date')

    # Median ADR per month
    fig, ax = plt.subplots(figsize=(5, 4))
    df.groupby([df['arrival_date'].dt.month, 'hotel'])['adr'].median().unstack().plot(ax=ax, marker='o', color=HOTEL_COLORS)
    ax.set(xlabel='Arrival Month', ylabel='Median ADR (currency units per night)', xticks=range(1, 13))
    _save(fig, '03_adr_per_month.png', 'Median ADR per Arrival Month')

    # ADR distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    for h, c in HOTEL_COLORS.items():
        ax.hist(df.loc[df['hotel'] == h, 'adr'], bins=50, alpha=0.6, label=h, color=c)
    ax.set(xlabel='ADR (currency units per night)', ylabel='Number of Bookings')
    ax.legend()
    _save(fig, '04_adr_distribution.png', 'ADR Distribution (Cleaned Data)')

    # Cancellation rate by lead time
    fig, ax = plt.subplots(figsize=(5, 4))
    lead = pd.cut(df['lead_time'], [-1, 7, 30, 90, 180, 365, 800], labels=['0-7', '8-30', '31-90', '91-180', '181-365', '366+'])
    (df.groupby([lead, 'hotel'], observed=True)['is_canceled'].mean().unstack() * 100).plot.bar(ax=ax, rot=0, color=HOTEL_COLORS)
    ax.axhline(df['is_canceled'].mean() * 100, linestyle='--', color='gray', label='All bookings')
    ax.set(xlabel='Lead Time (Days)', ylabel='Cancellation Rate (%)')
    ax.legend()
    _save(fig, '05_cancel_rate_by_lead_time.png', 'Cancellation Rate by Lead Time (All Cleaned Bookings)')

    # Empirical log-odds vs log lead time (model-selection training window)
    fig, ax = plt.subplots(figsize=(5, 4))
    for h, c in HOTEL_COLORS.items():
        sub = train[train['hotel'] == h]
        bins = pd.qcut(sub['lead_time'], 20, duplicates='drop')
        x = np.log1p(sub['lead_time']).groupby(bins, observed=True).mean()
        rate = sub['is_canceled'].groupby(bins, observed=True).mean()
        ax.plot(x, np.log(rate / (1 - rate)), 'o-', label=h, color=c)
    ax.set(xlabel='log(1 + Lead Time in Days)', ylabel='Empirical Log-Odds of Cancellation')
    ax.legend()
    _save(fig, '06_logodds_vs_lead_time.png', 'Log-Odds vs Lead Time (Selection Training Window)')

    # Drift of the previous-cancellation effect
    fig, ax = plt.subplots(figsize=(6, 4))
    quarter = refundable['arrival_date'].dt.to_period('Q').astype(str)
    rate = refundable.groupby([quarter, 'prev_cancel'])['is_canceled'].mean().unstack() * 100
    ax.plot(rate.index, rate[0], 'o-', label='No previous cancellation')
    ax.plot(rate.index, rate[1], 'o-', label='Has previous cancellation')
    for q, label in [('2016Q3', 'Validation start'), ('2017Q1', 'Test start')]:
        ax.axvline(list(rate.index).index(q) - 0.5, linestyle='--', color='gray')
        ax.text(list(rate.index).index(q) - 0.45, 95, label, fontsize=8, color='gray')
    ax.set(xlabel='Arrival Quarter', ylabel='Cancellation Rate (%)')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='center left')
    _save(fig, '07_prev_cancel_drift.png', 'Previous Cancellations: Effect Changes Over Time')


def temporal_design(df: pd.DataFrame) -> None:
    """Booking date vs arrival date, colored by the split each booking belongs to."""
    s = df.sample(8000, random_state=config.SEED)
    v, t = config.VALIDATION_START, config.TEST_START
    groups = {'Selection training (arrived before validation start)': s['arrival_date'] < v,
              'Validation (booked after validation start, arrived before test start)':
                  (s['booking_date'] >= v) & (s['arrival_date'] < t),
              'Not used in selection (booked before, arrived after validation start)':
                  (s['booking_date'] < v) & (s['arrival_date'] >= v) & (s['arrival_date'] < t),
              'Test (booked on or after test start)': s['booking_date'] >= t,
              'Never evaluated (booked before, arrived after test start)': (s['booking_date'] < t) & (s['arrival_date'] >= t)}
    colors = ['#1f77b4', '#2ca02c', '#bbbbbb', '#d62728', '#7f7f7f']
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for (label, m), c in zip(groups.items(), colors):
        ax.scatter(s.loc[m, 'booking_date'], s.loc[m, 'arrival_date'], s=3, alpha=0.5, color=c, label=label)
    for d in (v, t):
        ax.axvline(d, color='black', linewidth=0.8)
        ax.axhline(d, color='black', linewidth=0.8, linestyle='--')
    ax.set(xlabel='Booking Date (arrival date - lead time)', ylabel='Arrival Date')
    ax.legend(fontsize=7, markerscale=4, loc='upper left')
    _save(fig, '01_temporal_design.png', 'Temporal Design: Which Bookings Are Used Where (8,000-Row Sample)')


# ---------- Model performance ----------
def model_comparison(ci: pd.DataFrame) -> None:
    metrics = [('roc_auc', 'ROC-AUC (higher is better)'), ('average_precision', 'Average Precision (higher is better)'),
               ('log_loss', 'Log Loss (lower is better)'), ('brier', 'Brier Score (lower is better)')]
    models = list(dict.fromkeys(ci['model']))
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, (m, label) in zip(axes, metrics):
        d = ci[ci['metric'] == m].set_index('model').loc[models]
        y = np.arange(len(models))
        ax.errorbar(d['estimate'], y, xerr=[d['estimate'] - d['ci_lower_95'], d['ci_upper_95'] - d['estimate']],
                    fmt='o', capsize=3)
        ax.set_yticks(y, models)
        ax.set_title(label, fontsize=10)
    axes[0].invert_yaxis()
    _save(fig, '08_model_comparison.png', 'Test Performance with 95% Day-Clustered Bootstrap CIs')


def roc_pr(y, preds: dict) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.5))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        a1.plot(fpr, tpr, label=name)
        a2.plot(rec, prec, label=name)
    a1.plot([0, 1], [0, 1], '--', color='gray', label='Random')
    a2.axhline(np.mean(y), linestyle='--', color='gray', label=f'Random (rate = {np.mean(y):.2f})')
    a1.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC Curve')
    a2.set(xlabel='Recall', ylabel='Precision', title='Precision-Recall Curve')
    a1.legend(fontsize=8)
    a2.legend(fontsize=8)
    _save(fig, '09_roc_pr_curves.png', 'Discrimination on the Test Set')


def calibration(y, preds: dict) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, p in preds.items():
        t = calibration_table(y, p)
        se = np.sqrt(t['observed_rate'] * (1 - t['observed_rate']) / t['n'])
        ax.errorbar(t['mean_predicted'], t['observed_rate'], yerr=1.96 * se, fmt='o-', capsize=2, label=name)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], '--', color='gray', label='Perfect calibration')
    ax.set(xlabel='Mean Predicted Probability (Decile)', ylabel='Observed Cancellation Rate')
    ax.legend(fontsize=8)
    _save(fig, '10_calibration.png', 'Calibration by Decile (Test Set, 95% CI)')


def capture(curves: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, d in curves.groupby('model', sort=False):
        ax.plot(d['share_targeted'] * 100, d['share_captured'] * 100, label=name)
    ax.plot([0, 100], [0, 100], '--', color='gray', label='Random')
    for k in config.TOP_K:
        ax.axvline(k * 100, color='lightgray', linewidth=0.8)
    ax.set(xlabel='Bookings Targeted, Highest Risk First (%)', ylabel='Cancellations Captured (%)')
    ax.legend(fontsize=8)
    _save(fig, '11_risk_capture_curve.png', 'Risk Capture Curve (Test Set)')


# ---------- Interpretation ----------
def interaction(effects: pd.DataFrame, coefs: pd.DataFrame) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, h in [('city', 'City Hotel'), ('resort', 'Resort Hotel')]:
        a1.plot(effects['lead_time'], effects[f'prob_{key}'], label=h, color=HOTEL_COLORS[h])
        a1.fill_between(effects['lead_time'], effects[f'prob_{key}_lower_95'], effects[f'prob_{key}_upper_95'],
                        alpha=0.2, color=HOTEL_COLORS[h])
    a1.set(xlabel='Lead Time (Days)', ylabel='Average Predicted Cancellation Probability',
           title='Average Predicted Probability by Hotel\n(other covariates as in 3,000 training bookings)')
    a1.legend()
    a2.plot(effects['lead_time'], effects['odds_ratio'], color='black')
    a2.fill_between(effects['lead_time'], effects['or_lower_95'], effects['or_upper_95'], alpha=0.2, color='black')
    a2.axhline(1, linestyle='--', color='gray')
    a2.set(xlabel='Lead Time (Days)', ylabel='Odds Ratio, Resort vs City',
           title='Resort vs City Odds Ratio Depends on Lead Time')
    for ax in (a1, a2):
        ax.title.set_fontsize(10)
    _save(fig, '12_hotel_lead_interaction.png', 'Lead Time x Hotel Interaction (Strict Logistic Model, 95% CI)')


def _pretty(term: str) -> str:
    for a, b in [('np.log1p(lead_time)', 'log(1+lead)'), ('market_segment[T.', 'segment: '), ('customer_type[T.', 'customer: '),
                 ('agent_group[T.', 'agent: '), ('distribution_channel[T.', 'channel: '), ('reserved_room_type[T.', 'room: '),
                 ('C(month)[T.', 'month: '), ('[T.', ': '), (']', '')]:
        term = term.replace(a, b)
    return term


def odds_ratios(coefs: pd.DataFrame) -> None:
    d = coefs[coefs['term'] != 'Intercept']
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(5, 0.22 * len(d) + 1.5))
    ax.errorbar(d['odds_ratio'], y, xerr=[d['odds_ratio'] - d['or_lower_95'], d['or_upper_95'] - d['odds_ratio']],
                fmt='o', markersize=4, capsize=2)
    ax.axvline(1, linestyle='--', color='gray')
    ax.set_xscale('log')
    ax.minorticks_off()
    ax.set_xticks([0.2, 0.5, 1, 2, 5], ['0.2', '0.5', '1', '2', '5'])
    ax.set_yticks(y, [_pretty(t) for t in d['term']], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Odds Ratio (log scale), 95% CI')
    _save(fig, '13_odds_ratios.png', 'Conditional Odds Ratios\n(hotel and lead terms interact; see interaction figure)')


def subgroups(table: pd.DataFrame) -> None:
    t = table.copy()
    t['label'] = t['dimension'] + ': ' + t['group'] + ' (n=' + t['n'].astype(str) + ')' + np.where(t['unstable'], ' *', '')
    y = np.arange(len(t))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 0.25 * len(t) + 1.5), sharey=True)
    a1.plot(t['roc_auc'], y, 'o')
    a1.set(xlabel='ROC-AUC', title='Discrimination within Subgroup')
    a2.plot(t['actual_rate'], y, 'o', label='Actual')
    a2.plot(t['mean_predicted'], y, 'x', label='Predicted (mean)')
    a2.set(xlabel='Cancellation Rate', title='Calibration-in-the-Large')
    a2.legend(fontsize=8)
    a1.set_yticks(y, t['label'], fontsize=8)
    a1.invert_yaxis()
    for ax in (a1, a2):
        ax.title.set_fontsize(10)
    _save(fig, '14_subgroup_performance.png', 'Strict Logistic Model by Subgroup (Test Set; * = fewer than 50 events in a class)')


def daily(daily_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = {'actual': 'Actual'} | {c: f'Expected: {c}' for c in daily_df.columns if c != 'actual'}
    daily_df.rename(columns=labels).rolling(7).mean().plot(ax=ax, color=['black', '#d62728', '#7f7f7f'], style=['-', '-', '--'])
    ax.set(xlabel='Arrival Date', ylabel='Cancellations per Arrival Date (7-Day MA)')
    _save(fig, '15_daily_cancellations.png', 'Actual vs Expected Cancellations per Arrival Date (Test Bookings, Post-Hoc Grouping)')


def net_benefit(curve: pd.DataFrame, policy: pd.DataFrame) -> None:
    """Rows: expected (ex ante, primary) and outcome-conditioned (ex post) net benefit; columns: scenarios."""
    names = list(dict.fromkeys(curve['scenario']))
    views = [('expected_net_benefit', 'Expected net benefit\n(predicted probabilities)'),
             ('outcome_conditioned_net_benefit', 'Outcome-conditioned net benefit\n(observed test labels)')]
    fig, axes = plt.subplots(2, len(names), figsize=(5 * len(names), 7.5), sharex=True)
    test_policy = policy[policy['stage'] == 'test'].set_index('scenario')
    for col, s in enumerate(names):
        for row, (metric, label) in enumerate(views):
            ax = axes[row, col]
            for strategy, d in curve[curve['scenario'] == s].groupby('strategy', sort=False):
                style = '--' if strategy.startswith('Random') else '-'
                ax.plot(d['coverage'] * 100, d[metric] / 1000, style, label=strategy)
            pol = test_policy.loc[s]
            ax.plot(pol['coverage'] * 100, pol[metric] / 1000, 'k*', markersize=11,
                    label='Threshold policy: contact if expected profit > 0')
            ax.axhline(0, color='black', linewidth=0.8)
            ax.set(ylabel=f'{label} (thousands)', title=f'{s} scenario' if row == 0 else None)
            if row == 1:
                ax.set_xlabel('Intervention Coverage (% of Test Bookings, Highest Score First)')
    axes[0, 0].legend(fontsize=7, loc='lower left')
    _save(fig, '16_net_benefit_vs_coverage.png',
          'ILLUSTRATIVE scenario analysis: net benefit vs coverage under ASSUMED intervention parameters\n'
          '(currency units of ADR; no intervention was observed, neither row is realized profit)')
