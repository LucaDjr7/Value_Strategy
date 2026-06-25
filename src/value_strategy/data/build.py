"""Data-layer orchestration (Part 1).

``build_panel_from_wrds`` chains the WRDS extraction, the merge, the intangible
computation, the cleaning and the illiquidity step, then writes the panel to a
parquet cache (overwriting the existing one). ``load_cached_panel`` re-reads
the cache without going through WRDS.
"""

from __future__ import annotations

import pandas as pd

from .. import config
from . import cache, cleaning, intangibles, panel as panel_mod, wrds_loader


def build_panel_from_wrds(write_cache: bool = True):
    """Build the monthly panel (Part 1) and cache it immediately.

    The Part 4 inputs (ML CRSP, macro) are fetched separately by
    :func:`fetch_ml_inputs_from_wrds`, so that (a) the expensive panel is
    saved before any heavy query, and (b) the large intermediate DataFrames are
    released before the ML fetch (lower memory peak).

    Returns
    -------
    panel, df_comp
    """
    print("=" * 60)
    print("  PART 1 — PANEL CONSTRUCTION (WRDS)")
    print("=" * 60)

    db = wrds_loader.connect()
    try:
        df_comp = wrds_loader.load_compustat(db)
        df_crsp = wrds_loader.load_crsp(db)
        df_crsp = wrds_loader.apply_delistings(db, df_crsp)
        df_link = wrds_loader.load_ccm_link(db)

        panel = panel_mod.merge_panel(df_crsp, df_comp, df_link)
        # Release the heavy intermediates as soon as they are no longer needed
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

    print("-> Panel ready for Part 2 (cache written)")
    return panel, df_comp


def fetch_ml_inputs_from_wrds(write_cache: bool = True):
    """Fetch the Part 4 inputs (ML-dedicated CRSP + FRED macro).

    Dedicated WRDS connection, run after Part 1 to limit memory usage.

    Returns
    -------
    crsp_ml, macro
    """
    print("=" * 60)
    print("  PART 4 — LOADING ML INPUTS (WRDS)")
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
    """Fetch all data (panel + ML inputs), from cache or from WRDS.

    In WRDS mode, the panel is cached before the ML fetch; if the panel cache
    already exists, it is reused so Part 1 is not re-run.

    Returns
    -------
    panel, df_comp, crsp_ml, macro
    """
    if use_cache:
        return load_cached_panel()

    if config.PANEL_CACHE.exists() and config.COMPUSTAT_CACHE.exists():
        print("  (panel already cached — Part 1 not re-run)")
        panel = cache.load_frame(config.PANEL_CACHE)
        panel["date"] = pd.to_datetime(panel["date"])
        df_comp = cache.load_frame(config.COMPUSTAT_CACHE)
    else:
        panel, df_comp = build_panel_from_wrds()

    crsp_ml, macro = fetch_ml_inputs_from_wrds()
    return panel, df_comp, crsp_ml, macro


def load_cached_panel():
    """Re-read panel, Compustat, ML CRSP and macro from the cache (no WRDS)."""
    print("=" * 60)
    print("  PART 1 — READING PANEL (cache)")
    print("=" * 60)
    panel = cache.load_frame(config.PANEL_CACHE)
    panel["date"] = pd.to_datetime(panel["date"])
    df_comp = cache.load_frame(config.COMPUSTAT_CACHE)
    crsp_ml = cache.load_frame(config.CRSP_ML_CACHE)
    crsp_ml["date"] = pd.to_datetime(crsp_ml["date"])
    macro = cache.load_frame(config.MACRO_CACHE)
    macro["date"] = pd.to_datetime(macro["date"])
    return panel, df_comp, crsp_ml, macro
