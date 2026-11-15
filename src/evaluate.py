"""Evaluation: discrimination, probability accuracy, calibration, ranking (Top-K), subgroups and daily totals."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src import config


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Calibration-in-the-large (intercept with logit(p) as offset; 0 is ideal) and slope (1 is ideal)."""
    lp = logit(p)
    intercept = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    return float(intercept), float(slope)


def metrics(y, p, calibration: bool = True) -> dict:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    out = {'n': len(y), 'cancellations': int(y.sum()),
           'roc_auc': roc_auc_score(y, p), 'average_precision': average_precision_score(y, p),
           'log_loss': log_loss(y, p, labels=[0, 1]), 'brier': brier_score_loss(y, p),
           'mean_predicted': p.mean(), 'actual_rate': y.mean()}
    if calibration:
        if np.ptp(p) > 0:
            out['calibration_intercept'], out['calibration_slope'] = calibration_intercept_slope(y, p)
        else:
            out['calibration_intercept'], out['calibration_slope'] = float(logit(y.mean()) - logit(p[0])), np.nan
    return out


def bootstrap_metrics(y, preds: dict[str, np.ndarray], groups, reference: str,
                      n_boot: int = config.N_BOOTSTRAP, seed: int = config.SEED) -> pd.DataFrame:
    """95% CIs from a cluster bootstrap that resamples whole arrival dates.

    Bookings on the same arrival date (e.g. group blocks) are correlated, so days rather than rows are resampled.
    Also reports the difference of every model against the reference model.
    """
    y, groups = np.asarray(y), np.asarray(groups)
    codes, uniques = pd.factorize(groups)
    members = [np.flatnonzero(codes == g) for g in range(len(uniques))]
    rng = np.random.default_rng(seed)
    names = ['roc_auc', 'average_precision', 'log_loss', 'brier']
    funcs = {'roc_auc': roc_auc_score, 'average_precision': average_precision_score,
             'log_loss': lambda a, b: log_loss(a, b, labels=[0, 1]), 'brier': brier_score_loss}
    samples = {m: {k: [] for k in names} for m in preds}
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        for m, p in preds.items():
            for k in names:
                samples[m][k].append(funcs[k](y[idx], np.asarray(p)[idx]))
    rows = []
    for m, p in preds.items():
        point = metrics(y, p, calibration=False)
        for k in names:
            s = np.array(samples[m][k])
            d = s - np.array(samples[reference][k])
            rows.append({'model': m, 'metric': k, 'estimate': point[k],
                         'ci_lower_95': np.percentile(s, 2.5), 'ci_upper_95': np.percentile(s, 97.5),
                         f'diff_vs_{reference}': point[k] - metrics(y, preds[reference], calibration=False)[k],
                         'diff_ci_lower_95': np.percentile(d, 2.5), 'diff_ci_upper_95': np.percentile(d, 97.5)})
    return pd.DataFrame(rows)


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({'y': np.asarray(y), 'p': np.asarray(p)})
    df['bin'] = pd.qcut(df['p'].rank(method='first'), n_bins, labels=False) + 1
    return (df.groupby('bin').agg(n=('y', 'size'), mean_predicted=('p', 'mean'), observed_rate=('y', 'mean'))
              .reset_index())


# ---------- Ranking ----------
def rank_order(score, seed: int = config.SEED) -> np.ndarray:
    """Indices from highest to lowest score; ties are broken at random (matters for coarse baselines)."""
    score = np.asarray(score, dtype=float)
    tie_break = np.random.default_rng(seed).random(len(score))
    return np.lexsort((tie_break, -score))


def top_k(y, score, share: float) -> dict:
    """Precision, recall (capture rate) and lift among the share of bookings with the highest score."""
    y = np.asarray(y)
    if len(y) == 0 or y.sum() == 0:
        raise ValueError('top_k needs at least one booking and one cancellation')
    k = max(1, int(round(share * len(y))))
    hits = y[rank_order(score)[:k]].sum()
    precision = hits / k
    return {'k_share': share, 'k': k, 'precision': precision, 'recall': hits / y.sum(), 'lift': precision / y.mean()}


def top_k_table(y, scores: dict[str, np.ndarray], shares=config.TOP_K) -> pd.DataFrame:
    rows = [{'model': name, **top_k(y, s, share)} for name, s in scores.items() for share in shares]
    base = np.mean(y)
    rows += [{'model': 'Random (expected)', 'k_share': share, 'k': int(round(share * len(y))),
              'precision': base, 'recall': share, 'lift': 1.0} for share in shares]
    return pd.DataFrame(rows)


def capture_curve(y, score, n_points: int = 101) -> pd.DataFrame:
    y = np.asarray(y)
    captured = np.concatenate([[0], np.cumsum(y[rank_order(score)])]) / y.sum()
    share = np.linspace(0, 1, n_points)
    return pd.DataFrame({'share_targeted': share, 'share_captured': captured[np.round(share * len(y)).astype(int)]})


# ---------- Subgroups ----------
def subgroup_table(df: pd.DataFrame, p, by: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    y = df['is_canceled'].to_numpy()
    p = np.asarray(p)
    for dimension, labels in by.items():
        labels = np.asarray(labels.astype(str))
        for level in pd.unique(labels):
            m = labels == level
            n_events = int(y[m].sum())
            two_classes = 0 < n_events < m.sum()
            rows.append({'dimension': dimension, 'group': level, 'n': int(m.sum()), 'cancellations': n_events,
                         'actual_rate': y[m].mean(), 'mean_predicted': p[m].mean(),
                         'roc_auc': roc_auc_score(y[m], p[m]) if two_classes else np.nan,
                         'brier': brier_score_loss(y[m], p[m]),
                         'unstable': n_events < config.MIN_SUBGROUP_EVENTS or (m.sum() - n_events) < config.MIN_SUBGROUP_EVENTS})
    return pd.DataFrame(rows)


# ---------- Daily cancellation totals ----------
def daily_totals(df: pd.DataFrame, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    daily = df.assign(**{k: np.asarray(v) for k, v in preds.items()})
    return daily.groupby('arrival_date')[['is_canceled', *preds]].sum().rename(columns={'is_canceled': 'actual'})


def daily_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in daily.columns.drop('actual'):
        err = daily[col] - daily['actual']
        rows.append({'forecast': col, 'mae': err.abs().mean(), 'rmse': np.sqrt((err ** 2).mean()),
                     'bias': err.mean(), 'correlation': daily[col].corr(daily['actual'])})
    return pd.DataFrame(rows)
