"""Extraction des données brutes WRDS (CRSP, Compustat, delistings, lien CCM)
et macro FRED.

Toutes les requêtes SQL sont reproduites à l'identique du notebook d'origine.
Chaque fonction prend une connexion ``wrds.Connection`` ouverte (voir
:func:`connect`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Détection des dépendances optionnelles pour la macro FRED
# ----------------------------------------------------------------------------
try:
    from pandas_datareader import data as pdr
    HAS_PDR = True
except ImportError:
    HAS_PDR = False

try:
    from fredapi import Fred
    HAS_FREDAPI = True
except ImportError:
    HAS_FREDAPI = False


# ----------------------------------------------------------------------------
# 1.1  Connexion WRDS
# ----------------------------------------------------------------------------
def _load_env_local() -> dict:
    """Parse un fichier .env.local à la racine du projet (KEY=VALUE par ligne).

    Jamais loggé. Les guillemets entourant la valeur sont retirés.
    """
    from .. import config

    env_path = config.ROOT_DIR / ".env.local"
    values: dict = {}
    if not env_path.exists():
        return values
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        values[key.strip()] = val.strip().strip("'").strip('"')
    return values


def connect():
    """Ouvre une connexion WRDS.

    Cherche les identifiants dans cet ordre :
      1. fichier ``.env.local`` (WRDS_USERNAME / WRDS_PASSWORD),
      2. variables d'environnement (WRDS_USERNAME / WRDS_PASSWORD),
      3. sinon, demande interactive de la librairie ``wrds``.

    Le mot de passe n'est jamais affiché ni journalisé.
    """
    import os

    import wrds

    env = _load_env_local()
    username = env.get("WRDS_USERNAME") or os.environ.get("WRDS_USERNAME")
    password = env.get("WRDS_PASSWORD") or os.environ.get("WRDS_PASSWORD")

    if username and password:
        print("Connexion WRDS (identifiants depuis .env.local)...")
        return wrds.Connection(wrds_username=username, wrds_password=password)
    print("Connexion WRDS (identifiants interactifs)...")
    return wrds.Connection()


# ----------------------------------------------------------------------------
# 1.2  Compustat annuel
#
#   Firmes industrielles US (indfmt = INDL), 1997-2025. On démarre en 1997
#   pour avoir >= 3 ans d'historique avant 2000 afin d'initialiser les stocks
#   KC et OC. Filtres : sale > 0, at > 0, ceq > 0.
# ----------------------------------------------------------------------------
COMPUSTAT_QUERY = """
    SELECT
        gvkey, datadate, fyear, indfmt, conm,
        sale, revt, ebit, oibdp, dp, ni, txt, xint,
        oancf, capx, cogs,
        at, ceq, dltt, dlc, che, lct,
        csho, prcc_f,
        xrd, xsga
    FROM comp.funda
    WHERE fyear BETWEEN 1997 AND 2025
      AND indfmt  = 'INDL'
      AND datafmt = 'STD'
      AND popsrc  = 'D'
      AND consol  = 'C'
      AND sale > 0
      AND at   > 0
      AND ceq  > 0
"""


def load_compustat(db) -> pd.DataFrame:
    """Compustat annuel nettoyé (déduplication gvkey/fyear)."""
    df = db.raw_sql(COMPUSTAT_QUERY, date_cols=["datadate"])
    df = df.drop_duplicates(["gvkey", "fyear"])
    df = df.sort_values(["gvkey", "datadate"]).reset_index(drop=True)
    print(f"Compustat : {df.shape[0]:,} obs — {df['gvkey'].nunique():,} firmes")
    return df


# ----------------------------------------------------------------------------
# 1.3  CRSP mensuel
#
#   Actions ordinaires US (shrcd 10/11), NYSE/AMEX/NASDAQ (exchcd 1/2/3).
#   Volume dollar de juin et décembre (rebalancement semestriel).
# ----------------------------------------------------------------------------
CRSP_QUERY = """
    SELECT
        m.permno,
        m.date,
        m.prc,
        m.ret,
        m.retx,
        m.shrout,
        n.siccd,
        n.shrcd,
        n.exchcd,
        v.vol_monthly,
        v.vol_daily_avg,
        v.dollar_vol_monthly_m
    FROM crsp.msf AS m
    JOIN crsp.msenames AS n
        ON  m.permno = n.permno
        AND m.date BETWEEN n.namedt AND n.nameendt
    LEFT JOIN (
        SELECT
            permno,
            DATE_TRUNC('month', date)   AS date,
            SUM(vol)                     AS vol_monthly,
            AVG(vol)                     AS vol_daily_avg,
            SUM(vol * ABS(prc)) / 1e6   AS dollar_vol_monthly_m
        FROM crsp.dsf
        WHERE date >= '2000-01-01'
          AND EXTRACT(MONTH FROM date) IN (6, 12)
        GROUP BY permno, DATE_TRUNC('month', date)
    ) AS v
        ON  m.permno = v.permno
        AND DATE_TRUNC('month', m.date) = v.date
    WHERE n.shrcd  IN (10, 11)
      AND n.exchcd IN (1, 2, 3)
      AND m.date >= '2000-01-01'
"""


def load_crsp(db) -> pd.DataFrame:
    """CRSP mensuel (prix, rendements, volume dollar semestriel)."""
    df = db.raw_sql(CRSP_QUERY, date_cols=["date"])
    df["date"] = df["date"] + pd.offsets.MonthEnd(0)
    print(f"CRSP : {df.shape[0]:,} obs — {df['permno'].nunique():,} titres")
    return df


# ----------------------------------------------------------------------------
# 1.4  Delistings — correction du survivorship bias (Shumway 2001)
# ----------------------------------------------------------------------------
DELIST_QUERY = """
    SELECT permno, dlstdt, dlret, dlstcd, dlprc
    FROM crsp.msedelist
    WHERE dlstdt >= '2000-01-01'
"""


def apply_delistings(db, df_crsp: pd.DataFrame) -> pd.DataFrame:
    """Intègre les rendements de delisting dans le panel CRSP.

    a) Delistings directs : date delisting = date msf -> merge direct.
    b) Orphan delistings : delisting APRÈS la dernière observation msf
       (gap <= 6 mois) -> rattaché à la dernière date dispo.
    c) Nettoyage des rendements aberrants (ret < -1 -> NaN).
    """
    df_crsp = df_crsp.copy()
    db_dl = db.raw_sql(DELIST_QUERY, date_cols=["dlstdt"])
    db_dl["date"] = db_dl["dlstdt"] + pd.offsets.MonthEnd(0)

    # a) Delistings directs
    df_crsp = df_crsp.merge(
        db_dl[["permno", "date", "dlret", "dlstcd", "dlprc"]],
        on=["permno", "date"], how="left",
    )

    # b) Orphan delistings
    last_obs = df_crsp.groupby("permno")["date"].max().reset_index(name="last_date")
    orphan_dl = db_dl.merge(last_obs, on="permno")
    orphan_dl = orphan_dl[orphan_dl["date"] > orphan_dl["last_date"]].copy()
    orphan_dl["gap_months"] = (
        (orphan_dl["date"].dt.year - orphan_dl["last_date"].dt.year) * 12
        + (orphan_dl["date"].dt.month - orphan_dl["last_date"].dt.month)
    )
    orphan_valid = orphan_dl[orphan_dl["gap_months"] <= 6]

    if not orphan_valid.empty:
        df_crsp = df_crsp.merge(
            orphan_valid[["permno", "last_date", "dlret", "dlstcd", "dlprc"]]
            .rename(columns={
                "last_date": "date",
                "dlret": "dlret_orphan",
                "dlstcd": "dlstcd_orphan",
                "dlprc": "dlprc_orphan",
            }),
            on=["permno", "date"], how="left",
        )
        df_crsp["dlret"] = df_crsp["dlret"].fillna(df_crsp["dlret_orphan"])
        df_crsp["dlstcd"] = df_crsp["dlstcd"].fillna(df_crsp["dlstcd_orphan"])
        df_crsp["dlprc"] = df_crsp["dlprc"].fillna(df_crsp["dlprc_orphan"])
        df_crsp.drop(
            columns=["dlret_orphan", "dlstcd_orphan", "dlprc_orphan"], inplace=True,
        )

    # Nettoyage rendements aberrants
    df_crsp["ret"] = pd.to_numeric(df_crsp["ret"], errors="coerce")
    df_crsp.loc[df_crsp["ret"] < -1, "ret"] = np.nan
    return df_crsp


# ----------------------------------------------------------------------------
# 1.5  Table de lien CCM (gvkey <-> permno)
# ----------------------------------------------------------------------------
LINK_QUERY = """
    SELECT gvkey, lpermno AS permno, linkdt, linkenddt
    FROM crsp.ccmxpf_lnkhist
    WHERE linktype IN ('LC', 'LU')
      AND linkprim IN ('P', 'C')
"""


def load_ccm_link(db) -> pd.DataFrame:
    """Table de lien Compustat-CRSP (liens primaires valides)."""
    df = db.raw_sql(LINK_QUERY, date_cols=["linkdt", "linkenddt"])
    df["linkenddt"] = df["linkenddt"].fillna(pd.Timestamp("2099-12-31"))
    df["permno"] = df["permno"].astype(float)
    print(f"CCM liens : {df.shape[0]:,}")
    return df


# ----------------------------------------------------------------------------
# 4.0a  CRSP mensuel dédié au ML (volume natif pour TOUS les mois)
# ----------------------------------------------------------------------------
ML_CRSP_QUERY = """
    SELECT
        m.permno,
        m.date,
        m.ret,
        m.retx,
        m.prc,
        m.vol,
        m.shrout
    FROM crsp.msf AS m
    JOIN crsp.msenames AS n
        ON  m.permno = n.permno
        AND m.date BETWEEN n.namedt AND n.nameendt
    WHERE n.shrcd  IN (10, 11)
      AND n.exchcd IN (1, 2, 3)
      AND m.date >= '2000-01-01'
"""


def load_crsp_ml(db) -> pd.DataFrame:
    """CRSP mensuel avec volume natif tous les mois (features ML)."""
    df = db.raw_sql(ML_CRSP_QUERY, date_cols=["date"])
    df["date"] = df["date"] + pd.offsets.MonthEnd(0)
    vol_coverage = df.groupby("date")["vol"].apply(lambda x: x.notna().mean())
    print(f"  CRSP ML : {df.shape[0]:,} obs | {df['permno'].nunique():,} titres "
          f"| {df['date'].nunique()} mois")
    print(f"  Couverture vol : {vol_coverage.mean():.1%} "
          f"(min {vol_coverage.min():.1%}, max {vol_coverage.max():.1%})")
    return df


# ----------------------------------------------------------------------------
# 4.0c  Macro FRED (avec fallback synthétique si inaccessible)
# ----------------------------------------------------------------------------
def fetch_fred_macro(start: str = "2000-01-01", end: str = "2025-06-30") -> pd.DataFrame:
    """Récupère les features macro FRED. Fallback synthétique si inaccessible."""
    print("[4.0c] Fetching FRED macro data...")
    fred_series = {
        "VIXCLS": "vix",
        "BAMLC0A0CM": "credit_spread",
        "T10Y2Y": "term_spread",
        "TEDRATE": "ted_spread",
        "DFF": "fed_funds",
        "UMCSENT": "consumer_sent",
    }
    macro_df = None

    # Tentative 1 : pandas_datareader
    if HAS_PDR:
        try:
            frames = {}
            for fred_id, col_name in fred_series.items():
                try:
                    s = pdr.DataReader(fred_id, "fred", start, end)
                    s.columns = [col_name]
                    frames[col_name] = s
                except Exception as e:  # noqa: BLE001
                    print(f"    ⚠ {fred_id}: {e}")
            if frames:
                macro_df = pd.concat(frames.values(), axis=1)
                macro_df.index.name = "date"
                macro_df = macro_df.reset_index()
                print(f"    ✓ {len(frames)} séries via pandas_datareader")
        except Exception as e:  # noqa: BLE001
            print(f"    ⚠ pandas_datareader failed: {e}")

    # Tentative 2 : fredapi
    if macro_df is None and HAS_FREDAPI:
        try:
            fred = Fred()
            frames = {}
            for fred_id, col_name in fred_series.items():
                try:
                    frames[col_name] = fred.get_series(
                        fred_id, observation_start=start, observation_end=end,
                    )
                except Exception:  # noqa: BLE001
                    pass
            if frames:
                macro_df = pd.DataFrame(frames)
                macro_df.index.name = "date"
                macro_df = macro_df.reset_index()
        except Exception as e:  # noqa: BLE001
            print(f"    ⚠ fredapi failed: {e}")

    # Fallback synthétique
    if macro_df is None:
        print("    ⚠ Fallback synthétique")
        dates = pd.date_range(start, end, freq="B")
        np.random.seed(99)
        n = len(dates)
        vix_base = 15 + 5 * np.sin(np.linspace(0, 8 * np.pi, n))
        stress_mask = np.zeros(n)
        for i, d in enumerate(dates):
            if pd.Timestamp("2008-09") <= d <= pd.Timestamp("2009-03"):
                stress_mask[i] = 40
            elif pd.Timestamp("2020-03") <= d <= pd.Timestamp("2020-04"):
                stress_mask[i] = 50
            elif pd.Timestamp("2011-08") <= d <= pd.Timestamp("2011-10"):
                stress_mask[i] = 15
            elif pd.Timestamp("2022-06") <= d <= pd.Timestamp("2022-10"):
                stress_mask[i] = 10
        vix = np.clip(vix_base + stress_mask + np.random.normal(0, 2, n), 9, 80)
        macro_df = pd.DataFrame({
            "date": dates,
            "vix": vix,
            "credit_spread": 1.5 + 0.03 * vix + np.random.normal(0, 0.2, n),
            "term_spread": 1.5 * np.sin(np.linspace(0, 4 * np.pi, n))
                           + np.random.normal(0, 0.3, n),
            "ted_spread": 0.3 + 0.01 * vix + np.random.normal(0, 0.1, n),
            "fed_funds": np.clip(
                3 + 2 * np.sin(np.linspace(0, 3 * np.pi, n))
                + np.random.normal(0, 0.2, n), 0, 6),
            "consumer_sent": np.clip(
                80 - 0.5 * vix + np.random.normal(0, 5, n), 50, 110),
        })

    macro_df["date"] = pd.to_datetime(macro_df["date"])
    macro_df = macro_df.set_index("date").resample("ME").mean().reset_index()
    macro_df = macro_df.ffill().bfill()
    print(f"    Macro panel : {len(macro_df)} mois, cols : {list(macro_df.columns[1:])}")
    return macro_df
