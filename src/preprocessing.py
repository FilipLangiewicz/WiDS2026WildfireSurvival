"""Feature engineering and the preprocessing pipeline.

All engineered features are deterministic row-wise transforms (no fitted state),
so they can sit inside the sklearn Pipeline before the scaler without any risk of
leakage; only StandardScaler learns parameters, and it does so per CV fold.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

EPS = 1e-6
HIT_RADIUS_M = 5000.0  # event triggers within 5 km of the evacuation-zone centroid

# Raw cyclic integer columns and their periods.
CYCLIC = {"event_start_hour": 24, "event_start_dayofweek": 7, "event_start_month": 12}


def _cyclic_encode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, period in CYCLIC.items():
        if col in out.columns:
            ang = 2.0 * np.pi * out[col] / period
            out[f"{col}_sin"] = np.sin(ang)
            out[f"{col}_cos"] = np.cos(ang)
            out = out.drop(columns=[col])
    return out


def _interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def col(name):
        return out[name] if name in out.columns else pd.Series(0.0, index=out.index)

    dist = col("dist_min_ci_0_5h")
    closing = col("closing_speed_m_per_h")
    closing_abs = col("closing_speed_abs_m_per_h")

    # How quickly the fire is eating into its distance margin.
    out["approach_urgency"] = closing / (dist + EPS)
    # Rough hours to cover the remaining margin to the 5 km radius.
    margin = np.clip(dist - HIT_RADIUS_M, 0.0, None)
    out["est_hours_to_zone"] = np.where(closing > 0, margin / (closing + EPS), 1e4)
    out["est_hours_to_zone"] = np.clip(out["est_hours_to_zone"], 0.0, 1e4)
    # Closing speed that is actually aimed at the zone.
    out["closing_x_alignment"] = closing * col("alignment_abs")
    # Aggressive growth combined with active approach.
    out["growth_x_closing"] = col("area_growth_rate_ha_per_h") * closing_abs
    # Projected advance relative to the current gap.
    out["advance_ratio"] = col("projected_advance_m") / (dist + EPS)
    return out


def engineer_features(df: pd.DataFrame, use_interactions: bool = True) -> pd.DataFrame:
    out = _cyclic_encode(df)
    # spread bearing is already provided as sin/cos; drop the redundant raw degrees.
    if "spread_bearing_deg" in out.columns:
        out = out.drop(columns=["spread_bearing_deg"])
    if use_interactions:
        out = _interactions(out)
    return out


def build_preprocessor(use_interactions: bool = True, scale: bool = True) -> Pipeline:
    """Pipeline step: feature engineering followed by optional standardization."""
    fe = FunctionTransformer(
        engineer_features,
        kw_args={"use_interactions": use_interactions},
        validate=False,
    )
    steps = [("features", fe)]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def engineered_feature_names(sample: pd.DataFrame, use_interactions: bool = True) -> list[str]:
    return list(engineer_features(sample, use_interactions=use_interactions).columns)
