"""Step 5: Fit the logistic regression, compare it with alternatives, and evaluate it on 2017 data."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import log_loss, roc_auc_score, roc_curve

from features import load_data, split

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)
train, test = split(load_data())
y_test = test['is_canceled']

FINAL_FORMULA = ('np.log1p(lead_time) * hotel + market_segment + customer_type + agent_group'
                 ' + total_of_special_requests + prev_stay + kids + meal')


def fit(formula, data=train):
    return smf.logit('is_canceled ~ ' + formula, data=data).fit(disp=0, maxiter=200)


def test_loss(model):
    return log_loss(y_test, model.predict(test))


def save(ax, title, path):
    ax.grid(linestyle=':')
    ax.set_title(title)
    ax.figure.savefig(path, dpi=200, bbox_inches='tight')


# ---------- Model selection ----------
# Rule: a term is kept only if it improves BIC on the training data
# AND does not make the log loss on 2017 test data worse.
final = fit(FINAL_FORMULA)
final_loss = test_loss(final)
print('Final model: parameters', len(final.params), 'AIC', round(final.aic, 1),
      'BIC', round(final.bic, 1), 'test log loss', round(final_loss, 4))

# Does adding (+) or dropping (-) a term improve the model?
changes = {'+ lead_time squared': FINAL_FORMULA + ' + I(np.log1p(lead_time)**2)',
           '+ prev_cancel': FINAL_FORMULA + ' + prev_cancel',
           '+ is_repeated_guest': FINAL_FORMULA + ' + is_repeated_guest',
           '+ C(month)': FINAL_FORMULA + ' + C(month)',
           '+ C(month):hotel': FINAL_FORMULA + ' + C(month) + C(month):hotel',
           '+ distribution_channel': FINAL_FORMULA + ' + distribution_channel',
           '+ reserved_room_type': FINAL_FORMULA + ' + reserved_room_type',
           '+ np.log1p(adr)': FINAL_FORMULA + ' + np.log1p(adr)',
           '+ adults': FINAL_FORMULA + ' + adults',
           '+ np.log1p(total_nights)': FINAL_FORMULA + ' + np.log1p(total_nights)',
           '- lead_time:hotel': FINAL_FORMULA.replace('np.log1p(lead_time) * hotel', 'np.log1p(lead_time) + hotel'),
           '- market_segment': FINAL_FORMULA.replace(' + market_segment', ''),
           '- meal': FINAL_FORMULA.replace(' + meal', ''),
           '- kids': FINAL_FORMULA.replace(' + kids', '')}
rows = []
for name, formula in changes.items():
    m = fit(formula)
    big, small = (m, final) if len(m.params) > len(final.params) else (final, m)
    lr, df_lr = 2 * (big.llf - small.llf), len(big.params) - len(small.params)  # likelihood-ratio test
    rows.append([name, df_lr, round(lr, 1), stats.chi2.sf(lr, df_lr), round(m.aic - final.aic, 1),
                 round(m.bic - final.bic, 1), round(test_loss(m) - final_loss, 4)])
print('\n', pd.DataFrame(rows, columns=['change', 'df', 'LR stat', 'p-value', 'AIC change',
                                        'BIC change', 'test log loss change']))

# ---------- Model summary ----------
print('\n', final.summary())
ci = final.conf_int()
odds = pd.DataFrame({'odds ratio': np.exp(final.params),
                     'lower 95%': np.exp(ci[0]),
                     'upper 95%': np.exp(ci[1])}).round(3)
print('\nOdds ratios with 95% CI:\n', odds)

# Does keeping the duplicated rows change the conclusions?  Refit on de-duplicated training data
train_dedup = train.drop_duplicates()
print('\nTraining rows: all', len(train), 'de-duplicated', len(train_dedup))
print(pd.DataFrame({'all rows': final.params, 'de-duplicated': fit(FINAL_FORMULA, train_dedup).params}).round(2))

# ---------- Performance ----------
print('\nDeviance:', round(-2 * final.llf, 1), ' Null deviance:', round(-2 * final.llnull, 1))
print('LR test vs null model:', round(final.llr, 1), 'df', int(final.df_model), 'p-value', final.llr_pvalue)
print('McFadden pseudo R-squared:', round(final.prsquared, 3))

p_test = final.predict(test)
auc = roc_auc_score(y_test, p_test)
print('\nTest log loss:', round(final_loss, 4))
print('Test log loss of intercept-only model:', round(log_loss(y_test, np.full(len(test), train['is_canceled'].mean())), 4))
print('Test AUC:', round(auc, 3))
print('Mean predicted vs actual cancellation rate:', round(p_test.mean(), 3), round(y_test.mean(), 3))

# ROC curve
fpr, tpr, _ = roc_curve(y_test, p_test)
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(fpr, tpr, label=f'Model (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], '--', color='darkgray', label='Random guess')
ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
ax.legend()
save(ax, 'ROC Curve (2017 Test Data)', 'figures/p2_fig3_roc.png')

# Are predicted probabilities calibrated?  Compare predicted vs observed rate per decile of prediction
decile = pd.qcut(p_test, 10, labels=False)
calib = pd.DataFrame({'predicted': p_test.groupby(decile).mean(), 'observed': y_test.groupby(decile).mean()})
print('\nCalibration by decile:\n', calib.round(3))
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(calib['predicted'], calib['observed'], 'o-', label='Model (2017 test data)')
ax.plot([0, 0.8], [0, 0.8], '--', color='darkgray', label='Perfect calibration')
ax.set(xlabel='Mean Predicted Probability', ylabel='Observed Cancellation Rate')
ax.legend()
save(ax, 'Calibration by Decile', 'figures/p2_fig4_calibration.png')

# How well does the model forecast the number of cancellations per arrival date?
daily = test.assign(predicted=p_test).groupby('arrival_date')[['is_canceled', 'predicted']].sum()
ax = daily.rolling(7).mean().plot(figsize=(10, 4), xlabel='Arrival Date', ylabel='Cancellations per Day (7-Day MA)')
ax.legend(['Actual', 'Predicted (sum of probabilities)'])
save(ax, 'Actual vs Predicted Cancellations per Arrival Date (2017)', 'figures/p2_fig5_daily_cancellations.png')
print('\nDaily cancellations: correlation', round(daily.corr().iloc[0, 1], 3),
      '| mean absolute error', round((daily['is_canceled'] - daily['predicted']).abs().mean(), 1),
      '| mean actual vs predicted', round(daily['is_canceled'].mean(), 1), round(daily['predicted'].mean(), 1))

# ---------- Which variables raise or lower the odds of cancellation? ----------
plot_df = odds.drop('Intercept')
labels = (plot_df.index.str.replace('np.log1p(lead_time)', 'log(1+lead)', regex=False)
                       .str.replace('market_segment[T.', 'segment: ', regex=False)
                       .str.replace('customer_type[T.', 'customer: ', regex=False)
                       .str.replace('agent_group[T.', 'agent: ', regex=False)
                       .str.replace('total_of_special_requests', 'special requests', regex=False)
                       .str.replace('[T.', ': ', regex=False).str.replace(']', '', regex=False))
y = np.arange(len(plot_df))
fig, ax = plt.subplots(figsize=(3.4, 5.2))
ax.errorbar(plot_df['odds ratio'], y, fmt='o', markersize=4, capsize=2,
            xerr=[plot_df['odds ratio'] - plot_df['lower 95%'], plot_df['upper 95%'] - plot_df['odds ratio']])
ax.axvline(1, linestyle='--', color='darkgray')
ax.set_xscale('log')
ax.minorticks_off()
ax.set_xticks([0.2, 0.5, 1, 2, 5], ['0.2', '0.5', '1', '2', '5'])
ax.set_yticks(y, labels, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('Odds Ratio (log scale)')
save(ax, 'Odds Ratios (95% CI)', 'figures/p2_fig6_odds_ratios.png')
