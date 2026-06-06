"""Miscellaneous helpers: reproducibility, paths, and small numeric utilities."""
import json
import os
import random
import zipfile
from pathlib import Path

import joblib
import numpy as np

RANDOM_STATE = 42
HORIZONS = (12, 24, 48, 72)
OBS_WINDOW = 72  # full observation window in hours; event indicator is defined over it


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    """Seed Python and NumPy RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def project_root() -> Path:
    """Return the project root regardless of where a notebook is launched from."""
    here = Path(__file__).resolve()
    return here.parent.parent


def data_dir() -> Path:
    return project_root() / "data"


def storage_dir() -> Path:
    """Return the writable directory used for generated artifacts."""
    root = os.environ.get("WIDS_STORAGE_DIR")
    return Path(root).expanduser().resolve() if root else project_root()


def results_dir() -> Path:
    out = storage_dir() / "results"
    out.mkdir(parents=True, exist_ok=True)
    return out


def artifacts_dir() -> Path:
    out = storage_dir() / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_table(df, name: str) -> Path:
    """Persist a dataframe as CSV in results/ and return the path."""
    path = results_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_figure(fig, name: str) -> Path:
    """Persist a matplotlib figure in results/ and return the path."""
    path = results_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return path


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def save_model_bundle(
    name: str,
    models,
    features,
    metadata=None,
    base_dir: Path | None = None,
    zip_bundle: bool = True,
) -> dict[str, Path]:
    """Save fitted models plus lightweight metadata in one artifact directory."""
    out_dir = (base_dir or artifacts_dir()) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    models_path = out_dir / "models.joblib"
    features_path = out_dir / "features.json"
    metadata_path = out_dir / "metadata.json"

    joblib.dump(models, models_path)
    features_path.write_text(json.dumps(list(features), indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(_json_ready(metadata or {}), indent=2), encoding="utf-8")

    paths = {"dir": out_dir, "models": models_path, "features": features_path, "metadata": metadata_path}
    if zip_bundle:
        zip_path = out_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            for path in [models_path, features_path, metadata_path]:
                zf.write(path, path.name)
        paths["zip"] = zip_path
    return paths


def enforce_monotonic(prob_matrix: np.ndarray) -> np.ndarray:
    """Make per-horizon hit probabilities non-decreasing across horizons.

    Independent per-horizon classifiers can produce P(12h) > P(24h), which is
    impossible because the event sets are nested. A running maximum across the
    horizon axis restores monotonicity without distorting ordering.
    """
    return np.maximum.accumulate(np.asarray(prob_matrix, dtype=float), axis=1)
