"""Orchestration de la couche données (Partie 1).

``build_panel_from_wrds`` enchaîne l'extraction WRDS, le merge, le calcul des
intangibles, le nettoyage et l'illiquidité, puis écrit le panel en cache
parquet (en écrasant l'existant). ``load_cached_panel`` relit le cache sans
repasser par WRDS.
"""

from __future__ import annotations

import pandas as pd

from .. import config
from . import cache, cleaning, intangibles, panel as panel_mod, wrds_loader


def build_panel_from_wrds(write_cache: bool = True):
    """Construit le panel mensuel (Partie 1) et le met en cache immédiatement.

    Les entrées de la Partie 4 (CRSP ML, macro) sont récupérées séparément par
    :func:`fetch_ml_inputs_from_wrds`, afin que (a) le panel coûteux soit
    sauvegardé avant toute requête lourde, et (b) les gros DataFrames
    intermédiaires soient libérés avant le fetch ML (réduction du pic mémoire).

    Returns
    -------
    panel, df_comp
    """
    print("=" * 60)
    print("  PARTIE 1 — CONSTRUCTION DU PANEL (WRDS)")
    print("=" * 60)

    db = wrds_loader.connect()
    try:
        df_comp = wrds_loader.load_compustat(db)
        df_crsp = wrds_loader.load_crsp(db)
        df_crsp = wrds_loader.apply_delistings(db, df_crsp)
        df_link = wrds_loader.load_ccm_link(db)

        panel = panel_mod.merge_panel(df_crsp, df_comp, df_link)
        # Libère les intermédiaires lourds dès qu'ils ne servent plus
        del df_crsp, df_link
        panel = intangibles.add_intangibles(panel)
        panel = cleaning.clean_variables(panel)
        panel = cleaning.add_illiquidity_costs(panel)
    finally:
        try:
            db.close()
        except Exception:  # noqa: BLE001
            pass

    if write_cache:
        cache.save_frame(panel, config.PANEL_CACHE)
        cache.save_frame(df_comp, config.COMPUSTAT_CACHE)

    print("→ Panel prêt pour la Partie 2 (cache écrit)")
    return panel, df_comp


def fetch_ml_inputs_from_wrds(write_cache: bool = True):
    """Récupère les entrées de la Partie 4 (CRSP dédié ML + macro FRED).

    Connexion WRDS dédiée, exécutée après la Partie 1 pour limiter la mémoire.

    Returns
    -------
    crsp_ml, macro
    """
    print("=" * 60)
    print("  PARTIE 4 — CHARGEMENT DES ENTRÉES ML (WRDS)")
    print("=" * 60)

    db = wrds_loader.connect()
    try:
        crsp_ml = wrds_loader.load_crsp_ml(db)
        macro = wrds_loader.fetch_fred_macro(
            start=str(crsp_ml["date"].min().date()),
            end=str(crsp_ml["date"].max().date()),
        )
    finally:
        try:
            db.close()
        except Exception:  # noqa: BLE001
            pass

    if write_cache:
        cache.save_frame(crsp_ml, config.CRSP_ML_CACHE)
        cache.save_frame(macro, config.MACRO_CACHE)
    return crsp_ml, macro


def acquire_data(use_cache: bool = False):
    """Récupère toutes les données (panel + entrées ML), du cache ou de WRDS.

    En mode WRDS, le panel est mis en cache avant le fetch ML ; si le cache du
    panel existe déjà, il est réutilisé pour ne pas relancer la Partie 1.

    Returns
    -------
    panel, df_comp, crsp_ml, macro
    """
    if use_cache:
        return load_cached_panel()

    if config.PANEL_CACHE.exists() and config.COMPUSTAT_CACHE.exists():
        print("  (panel déjà en cache — Partie 1 non relancée)")
        panel = cache.load_frame(config.PANEL_CACHE)
        panel["date"] = pd.to_datetime(panel["date"])
        df_comp = cache.load_frame(config.COMPUSTAT_CACHE)
    else:
        panel, df_comp = build_panel_from_wrds()

    crsp_ml, macro = fetch_ml_inputs_from_wrds()
    return panel, df_comp, crsp_ml, macro


def load_cached_panel():
    """Relit le panel, Compustat, CRSP ML et macro depuis le cache (sans WRDS)."""
    print("=" * 60)
    print("  PARTIE 1 — LECTURE DU PANEL (cache)")
    print("=" * 60)
    panel = cache.load_frame(config.PANEL_CACHE)
    panel["date"] = pd.to_datetime(panel["date"])
    df_comp = cache.load_frame(config.COMPUSTAT_CACHE)
    crsp_ml = cache.load_frame(config.CRSP_ML_CACHE)
    crsp_ml["date"] = pd.to_datetime(crsp_ml["date"])
    macro = cache.load_frame(config.MACRO_CACHE)
    macro["date"] = pd.to_datetime(macro["date"])
    return panel, df_comp, crsp_ml, macro
