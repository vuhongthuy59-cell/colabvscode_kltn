from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


SPLITS = ("train", "validation", "test")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Return MAE and RMSE with the same formula used across model scripts."""
    return (
        float(mean_absolute_error(y_true, y_pred)),
        float(np.sqrt(mean_squared_error(y_true, y_pred))),
    )


def split_snapshot_ids(index: pd.DataFrame) -> tuple[set[int], set[int], set[int]]:
    """Temporal 70/15/15 split by event date to avoid look-ahead leakage."""
    ordered = index.sort_values(["event_trading_date", "snapshot_id"]).reset_index(drop=True)
    unique_dates = ordered["event_trading_date"].drop_duplicates().reset_index(drop=True)
    train_end = int(len(unique_dates) * 0.70)
    val_end = int(len(unique_dates) * 0.85)
    train_dates = set(unique_dates.iloc[:train_end])
    val_dates = set(unique_dates.iloc[train_end:val_end])
    test_dates = set(unique_dates.iloc[val_end:])
    return (
        set(ordered.loc[ordered["event_trading_date"].isin(train_dates), "snapshot_id"].astype(int)),
        set(ordered.loc[ordered["event_trading_date"].isin(val_dates), "snapshot_id"].astype(int)),
        set(ordered.loc[ordered["event_trading_date"].isin(test_dates), "snapshot_id"].astype(int)),
    )


def assign_temporal_split(samples: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Fill the split column using the shared temporal split definition."""
    train_ids, val_ids, test_ids = split_snapshot_ids(index)
    samples = samples.copy()
    samples.loc[samples["snapshot_id"].isin(train_ids), "split"] = "train"
    samples.loc[samples["snapshot_id"].isin(val_ids), "split"] = "validation"
    samples.loc[samples["snapshot_id"].isin(test_ids), "split"] = "test"
    return samples


def split_masks(samples: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(samples["split"].eq(split_name).to_numpy() for split_name in SPLITS)


def split_indices(samples: pd.DataFrame) -> dict[str, np.ndarray]:
    return {split_name: np.where(samples["split"].eq(split_name).to_numpy())[0] for split_name in SPLITS}
