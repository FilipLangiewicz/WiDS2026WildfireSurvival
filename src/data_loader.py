"""Data loading and the survival-to-classification label reformulation.

The competition is a survival task. For the standard-classification track we turn
it into one binary problem per horizon H. An observation is usable for horizon H
only if its status at time H is known:

    y = 1   if event == 1 and time_to_hit_hours <= H        (hit by H)
    y = 0   if time_to_hit_hours >= H                        (followed past H, no hit)
    drop    if event == 0 and time_to_hit_hours <  H         (censored before H)

For H equal to the full observation window (72h) the `event` column already
encodes the outcome ("0 = never hit within 72h"), so every row is usable and the
label is simply `event`.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import OBS_WINDOW, data_dir

ID_COL = "event_id"
TIME_COL = "time_to_hit_hours"
EVENT_COL = "event"


def load_raw(folder: Path | str | None = None):
    """Load train, test, sample submission and metadata as dataframes."""
    folder = Path(folder) if folder is not None else data_dir()
    train = pd.read_csv(folder / "train.csv")
    test = pd.read_csv(folder / "test.csv")
    submission = pd.read_csv(folder / "sample_submission.csv")
    meta = pd.read_csv(folder / "metaData.csv")
    return train, test, submission, meta


def feature_columns(train: pd.DataFrame) -> list[str]:
    """All raw predictor columns (everything except id and the two targets)."""
    drop = {ID_COL, TIME_COL, EVENT_COL}
    return [c for c in train.columns if c not in drop]


def horizon_labels(train: pd.DataFrame, horizon: int):
    """Return (usable_mask, y) for a single horizon following the censoring rule."""
    time = train[TIME_COL].to_numpy()
    event = train[EVENT_COL].to_numpy().astype(int)

    if horizon >= OBS_WINDOW:
        usable = np.ones(len(train), dtype=bool)
        y = (event == 1).astype(int)
        return usable, y

    hit_by_h = (event == 1) & (time <= horizon)
    observed_past_h = time >= horizon  # status at H known, no hit yet
    usable = hit_by_h | observed_past_h
    y = hit_by_h.astype(int)
    return usable, y


def make_horizon_dataset(train: pd.DataFrame, horizon: int, features: list[str]):
    """Return (X, y) restricted to rows whose status at `horizon` is known."""
    usable, y = horizon_labels(train, horizon)
    X = train.loc[usable, features].reset_index(drop=True)
    y = pd.Series(y[usable], name=f"y_{horizon}h").reset_index(drop=True)
    return X, y


def label_summary(train: pd.DataFrame, horizons) -> pd.DataFrame:
    """Tabulate positives / negatives / excluded counts per horizon."""
    rows = []
    for h in horizons:
        usable, y = horizon_labels(train, h)
        rows.append(
            {
                "horizon_h": h,
                "positives": int(y[usable].sum()),
                "negatives": int((usable).sum() - y[usable].sum()),
                "excluded": int((~usable).sum()),
                "usable": int(usable.sum()),
            }
        )
    return pd.DataFrame(rows)
