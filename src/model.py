"""Logistic regression, model selection on the validation window, baselines and a boosting benchmark."""
import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

from src import config

EPS = 1e-6


def formula(terms: list[str]) -> str:
    return 'is_canceled ~ ' + ' + '.join(terms)


def fit_logit(terms: list[str] | str, data: pd.DataFrame):
    f = terms if isinstance(terms, str) else formula(terms)
    if not f.startswith('is_canceled'):
        f = 'is_canceled ~ ' + f
    return smf.logit(f, data=data).fit(disp=0, maxiter=200)


def predict(model, data: pd.DataFrame) -> np.ndarray:
    return np.clip(np.asarray(model.predict(data)), EPS, 1 - EPS)


def drop_term(terms: list[str], term: str) -> list[str]:
    """Remove a term; dropping 'a * b' keeps the main effects 'a + b'."""
    if '*' in term:
        return [t if t != term else t.replace(' * ', ' + ') for t in terms]
    return [t for t in terms if t != term]


# ---------- Model selection (training + validation only, never the test set) ----------
def select_terms(train: pd.DataFrame, validation: pd.DataFrame, base_terms: list[str],
                 candidates: list[str]) -> tuple[list[str], pd.DataFrame]:
    """Greedy forward selection, then one backward pass.

    A change is accepted only if it lowers BIC on the training window AND lowers log loss on the
    validation window. The test set is never used here.
    """
    def score(terms):
        m = fit_logit(terms, train)
        return m, log_loss(validation['is_canceled'], predict(m, validation))

    terms = list(base_terms)
    current, current_loss = score(terms)
    log = [{'step': 0, 'action': 'start', 'term': '', 'n_params': len(current.params), 'bic': current.bic,
            'validation_log_loss': current_loss, 'delta_bic': 0.0, 'delta_validation_log_loss': 0.0, 'accepted': True,
            'converged': current.mle_retvals['converged']}]

    def try_change(step, action, term, new_terms):
        m, loss = score(new_terms)
        row = {'step': step, 'action': action, 'term': term, 'n_params': len(m.params), 'bic': m.bic,
               'validation_log_loss': loss, 'delta_bic': m.bic - current.bic,
               'delta_validation_log_loss': loss - current_loss, 'accepted': False, 'converged': m.mle_retvals['converged']}
        return row, m, loss

    step, remaining = 1, list(candidates)
    while remaining:
        trials = [try_change(step, 'add', t, terms + [t]) for t in remaining]
        ok = [tr for tr in trials if tr[0]['delta_bic'] < 0 and tr[0]['delta_validation_log_loss'] < 0]
        if ok:
            best = min(ok, key=lambda tr: tr[0]['delta_validation_log_loss'])
            best[0]['accepted'] = True
            terms.append(best[0]['term'])
            remaining.remove(best[0]['term'])
            current, current_loss = best[1], best[2]
        log += [tr[0] for tr in trials]
        step += 1
        if not ok:
            break

    for t in list(terms):  # backward pass: drop a term if both criteria improve
        row, m, loss = try_change(step, 'drop', t, drop_term(terms, t))
        if row['delta_bic'] < 0 and row['delta_validation_log_loss'] < 0:
            row['accepted'] = True
            terms = drop_term(terms, t)
            current, current_loss = m, loss
        log.append(row)
    return terms, pd.DataFrame(log)


# ---------- Baselines ----------
def overall_rate(train: pd.DataFrame) -> float:
    return float(train['is_canceled'].mean())


def hotel_rates(train: pd.DataFrame) -> dict:
    return train.groupby('hotel', observed=True)['is_canceled'].mean().to_dict()


def predict_hotel_rate(rates: dict, data: pd.DataFrame, default: float) -> np.ndarray:
    return data['hotel'].astype(str).map(rates).fillna(default).to_numpy(dtype=float)


def predict_scheme_c(model, nr_rate: float, data: pd.DataFrame) -> np.ndarray:
    """Segmented scheme: Non Refund bookings get their training cancellation rate, others the logistic model."""
    p = np.full(len(data), nr_rate)
    other = data['is_non_refund'].to_numpy() == 0
    if other.any():
        p[other] = predict(model, data[other])
    return p


# ---------- Gradient boosting benchmark ----------
def gbm_frame(data: pd.DataFrame) -> pd.DataFrame:
    return data[config.GBM_COLUMNS]


def fit_gbm(train: pd.DataFrame, params: dict) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(max_iter=300, categorical_features='from_dtype',
                                           random_state=config.SEED, **params)
    return model.fit(gbm_frame(train), train['is_canceled'])


def predict_gbm(model, data: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict_proba(gbm_frame(data))[:, 1], EPS, 1 - EPS)


def tune_gbm(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rows = []
    for params in config.GBM_GRID:
        loss = log_loss(validation['is_canceled'], predict_gbm(fit_gbm(train, params), validation))
        rows.append({**params, 'validation_log_loss': loss})
    table = pd.DataFrame(rows)
    best = table.loc[table['validation_log_loss'].idxmin(), ['learning_rate', 'max_leaf_nodes']].to_dict()
    best['max_leaf_nodes'] = int(best['max_leaf_nodes'])
    return best, table


# ---------- Interpretation ----------
def coefficient_table(model) -> pd.DataFrame:
    ci = model.conf_int()
    return pd.DataFrame({'coef': model.params, 'std_err': model.bse, 'p_value': model.pvalues,
                         'odds_ratio': np.exp(model.params), 'or_lower_95': np.exp(ci[0]),
                         'or_upper_95': np.exp(ci[1])}).rename_axis('term').reset_index()


def design(model, data: pd.DataFrame) -> np.ndarray:
    """Design matrix of new data with the fitted model's terms (statsmodels 0.14: design_info, 0.15: model_spec)."""
    info = getattr(model.model.data, 'design_info', None) or model.model.data.model_spec
    return np.asarray(patsy.build_design_matrices([info], data)[0])


def hotel_lead_effects(model, reference: pd.DataFrame, lead_grid: np.ndarray, n_draws: int = 500,
                       seed: int = config.SEED) -> pd.DataFrame:
    """Resort vs City Hotel effect at each lead time, and average predicted probabilities.

    odds_ratio: conditional odds ratio Resort vs City at that lead time, other covariates held fixed
    (exact Wald CI from the coefficient covariance). prob_*: predicted probability averaged over the
    reference bookings after setting their hotel and lead time (marginal standardization); the CI comes
    from draws of the coefficients from their estimated normal distribution.
    """
    beta, cov = model.params.to_numpy(), model.cov_params().to_numpy()
    draws = np.random.default_rng(seed).multivariate_normal(beta, cov, size=n_draws)
    hotels = reference['hotel'].cat.categories
    rows = []
    for lead in lead_grid:
        x = {h: design(model, reference.assign(lead_time=lead, hotel=pd.Categorical([h] * len(reference), categories=hotels)))
             for h in ['City Hotel', 'Resort Hotel']}
        contrast = x['Resort Hotel'][0] - x['City Hotel'][0]   # same for every row unless hotel interacts with other terms
        log_or, se = contrast @ beta, np.sqrt(contrast @ cov @ contrast)
        row = {'lead_time': lead, 'odds_ratio': np.exp(log_or),
               'or_lower_95': np.exp(log_or - 1.96 * se), 'or_upper_95': np.exp(log_or + 1.96 * se)}
        for h, xh in x.items():
            key = 'city' if h == 'City Hotel' else 'resort'
            sims = (1 / (1 + np.exp(-xh @ draws.T))).mean(axis=0)
            row[f'prob_{key}'] = (1 / (1 + np.exp(-xh @ beta))).mean()
            row[f'prob_{key}_lower_95'], row[f'prob_{key}_upper_95'] = np.percentile(sims, [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows)
