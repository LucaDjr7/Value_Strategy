"""Parquet cache for the panel and the raw data.

Reproducibility strategy: on each run with a WRDS connection, the panel is
re-downloaded then **overwritten** on disk (always the same starting point).
Subsequent runs can start from the cache without WRDS via the ``--use-cache``
flag of ``main.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config


def save_frame(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to parquet (overwriting the existing one)."""
    config.ensure_dirs()
    df.to_parquet(path, index=False)
    print(f"  -> cache written: {path.relative_to(config.ROOT_DIR)} "
          f"({len(df):,} rows)")


def load_frame(path: Path) -> pd.DataFrame:
    """Load a parquet DataFrame from the cache."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cache: {path}. Run the pipeline with a WRDS connection "
            f"at least once (without --use-cache)."
        )
    df = pd.read_parquet(path)
    print(f"  -> cache read: {path.relative_to(config.ROOT_DIR)} "
          f"({len(df):,} rows)")
    return df


def cache_exists() -> bool:
    """True if the main panel is already cached."""
    return config.PANEL_CACHE.exists()
