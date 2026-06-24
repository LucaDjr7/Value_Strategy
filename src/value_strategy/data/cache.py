"""Cache parquet du panel et des données brutes.

Stratégie de reproductibilité : à chaque exécution avec connexion WRDS, le
panel est re-téléchargé puis **écrasé** sur disque (toujours le même point de
départ). Les exécutions suivantes peuvent repartir du cache sans WRDS via le
flag ``--use-cache`` de ``main.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config


def save_frame(df: pd.DataFrame, path: Path) -> None:
    """Sauvegarde un DataFrame en parquet (écrase l'existant)."""
    config.ensure_dirs()
    df.to_parquet(path, index=False)
    print(f"  → cache écrit : {path.relative_to(config.ROOT_DIR)} "
          f"({len(df):,} lignes)")


def load_frame(path: Path) -> pd.DataFrame:
    """Charge un DataFrame parquet depuis le cache."""
    if not path.exists():
        raise FileNotFoundError(
            f"Cache absent : {path}. Lancez le pipeline avec une connexion "
            f"WRDS au moins une fois (sans --use-cache)."
        )
    df = pd.read_parquet(path)
    print(f"  → cache lu : {path.relative_to(config.ROOT_DIR)} "
          f"({len(df):,} lignes)")
    return df


def cache_exists() -> bool:
    """Vrai si le panel principal est déjà en cache."""
    return config.PANEL_CACHE.exists()
