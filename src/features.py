"""Derived features, temporal splits and a categorical encoder learned on the training window only."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.clean import booking_date


def load_clean(path: Path = config.CLEAN_DATA) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['arrival_date', 'reservation_status_date'])
    return add_features(df)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features. None of them uses information from other rows, so they cannot leak."""
    df = df.copy()
    df['booking_date'] = booking_date(df['arrival_date'], df['lead_time'])
    df['month'] = df['arrival_date'].dt.month
    df['prev_cancel'] = (df['previous_cancellations'] > 0).astype(int)
    df['prev_stay'] = (df['previous_bookings_not_canceled'] > 0).astype(int)
    df['kids'] = ((df['children'] + df['babies']) > 0).astype(int)
    df['weekend_share'] = df['stays_in_weekend_nights'] / df['total_nights']
    df['has_company_id'] = df['company'].notna().astype(int)
    df['is_non_refund'] = (df['deposit_type'] == 'Non Refund').astype(int)
    df['booking_value'] = df['adr'] * df['total_nights']   # room revenue, not profit
    return df


# ---------- Temporal splits ----------
def trained_window(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Bookings whose outcome is known on the cutoff date: they arrived before it."""
    return df[df['arrival_date'] < cutoff]


def booked_window(df: pd.DataFrame, start: pd.Timestamp, outcome_known_by: pd.Timestamp | None = None) -> pd.DataFrame:
    """Bookings made on or after start, optionally only those whose outcome is known by a later date."""
    mask = df['booking_date'] >= start
    if outcome_known_by is not None:
        mask &= df['arrival_date'] < outcome_known_by
    return df[mask]


@dataclass
class Splits:
    train: pd.DataFrame          # model selection: fit
    validation: pd.DataFrame     # model selection: compare candidates
    refit: pd.DataFrame          # final fit (all outcomes known at TEST_START)
    test: pd.DataFrame           # final, one-time evaluation
    legacy_test: pd.DataFrame    # the original project's test set (2017 arrivals), for comparison only


def make_splits(df: pd.DataFrame, validation_start: pd.Timestamp = config.VALIDATION_START,
                test_start: pd.Timestamp = config.TEST_START) -> Splits:
    return Splits(train=trained_window(df, validation_start),
                  validation=booked_window(df, validation_start, outcome_known_by=test_start),
                  refit=trained_window(df, test_start),
                  test=booked_window(df, test_start),
                  legacy_test=df[df['arrival_date'] >= test_start])


def split_summary(splits: Splits) -> pd.DataFrame:
    rows = []
    for name in ['train', 'validation', 'refit', 'test', 'legacy_test']:
        d = getattr(splits, name)
        rows.append({'split': name, 'rows': len(d),
                     'booking_date_min': d['booking_date'].min().date(), 'booking_date_max': d['booking_date'].max().date(),
                     'arrival_date_min': d['arrival_date'].min().date(), 'arrival_date_max': d['arrival_date'].max().date(),
                     'max_lead_time': int(d['lead_time'].max()), 'cancel_rate': round(d['is_canceled'].mean(), 3),
                     'non_refund_share': round(d['is_non_refund'].mean(), 3)})
    return pd.DataFrame(rows)


# ---------- Categorical encoding ----------
@dataclass
class Encoder:
    levels: dict[str, list[str]]   # column -> allowed levels, reference level first
    top_agents: list[str]


def agent_group(agent: pd.Series, top_agents: list[str]) -> pd.Series:
    ids = agent.astype('Int64').astype(str)
    return pd.Series(np.where(agent.isna(), 'None', np.where(ids.isin(top_agents), ids, 'Other')), index=agent.index)


def fit_encoder(train: pd.DataFrame, columns: list[str] = config.CATEGORICAL_COLUMNS,
                top_n_agents: int = config.TOP_AGENTS, min_count: int = config.MIN_CATEGORY_COUNT) -> Encoder:
    """Learn category levels from the training rows only."""
    top_agents = train['agent'].dropna().astype('Int64').astype(str).value_counts().head(top_n_agents).index.tolist()
    data = train.assign(agent_group=agent_group(train['agent'], top_agents))
    levels = {}
    for col in columns:
        counts = data[col].astype(str).value_counts()
        kept = counts[counts >= min_count].index.tolist()
        if len(kept) < len(counts) and 'Other' not in kept:
            kept.append('Other')
        reference = config.REFERENCE_LEVELS.get(col)
        if reference not in kept:
            reference = counts.index[0]
        levels[col] = [reference] + sorted(set(kept) - {reference})
    return Encoder(levels=levels, top_agents=top_agents)


def encode(df: pd.DataFrame, encoder: Encoder) -> pd.DataFrame:
    """Apply training levels. Rare or unseen levels become 'Other' (or the reference level if there is no 'Other')."""
    df = df.assign(agent_group=agent_group(df['agent'], encoder.top_agents))
    for col, levels in encoder.levels.items():
        values = df[col].astype(str)
        fallback = 'Other' if 'Other' in levels else levels[0]
        df[col] = pd.Categorical(values.where(values.isin(levels), fallback), categories=levels)
    return df
