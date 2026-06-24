# Restructuration du repo Value-Strategy — Design

**Date** : 2026-06-24
**Objectif** : transformer un notebook monolithique (`Value_Strategy_GARP.ipynb`, ~3500 lignes
de code, 48 cellules) en un repo Python structuré, lisible et présentable — vitrine portfolio/CV.

## Contexte

Stratégie long/short equity US Small/Mid Cap. Signal value = B/M ajusté intangibles
(KC + OC, Peters & Taylor 2017), ranké par secteur, filtré qualité + momentum. Short = bottom
qualité du bucket growth. Détection ML de régimes « junk rally » (HMM/GMM + walk-forward) pour
réduire dynamiquement le short. Données : CRSP + Compustat via **WRDS** (base payante, connexion
en ligne) + Fama-French + FRED.

Le notebook tourne aujourd'hui en *fetch WRDS live* à chaque exécution, avec un état partagé via
variables globales (`panel`, `panel_is`, `perf_is`, `perf_oos`, `mkt_df`, `results_df`...).

## Décisions validées

1. **Données** : *fetch WRDS → cache parquet (écrase le précédent à chaque run) → étapes
   suivantes lisent le cache.* Point de départ reproductible. Un flag `--use-cache` permet de
   relancer sans WRDS à partir du dernier parquet.
2. **Objectif** : vitrine portfolio/CV → priorité lisibilité, README soigné, structure claire,
   code propre. Tests légers (logique pure uniquement).
3. **Orchestration** : `main.py` par étapes **ET** notebook narratif fin qui importe les modules.

## Architecture

```
value-strategy/
├── README.md                  # pitch stratégie, résultats, schéma pipeline, how-to-run
├── requirements.txt
├── pyproject.toml             # métadonnées package (léger)
├── .gitignore                 # exclut cache data, .ipynb_checkpoints, anaconda_projects…
├── main.py                    # orchestrateur : data → signaux → backtest → ML → figures
├── src/value_strategy/
│   ├── __init__.py
│   ├── config.py              # params FIXES, chemins, dates IS/OOS, seuils
│   ├── data/
│   │   ├── wrds_loader.py      # connexion + requêtes Compustat / CRSP / delistings / CCM / FRED
│   │   ├── panel.py           # merge CRSP+Compustat → panel mensuel (anti look-ahead)
│   │   ├── intangibles.py     # KC/OC Peters & Taylor (2017)
│   │   ├── cleaning.py        # nettoyage variables + illiquidité Amihud / coûts dynamiques
│   │   └── cache.py           # save/load parquet (écrase à chaque run)
│   ├── signals.py             # value + neutralisation sectorielle + qualité + momentum + score
│   ├── portfolio.py           # construction L/S + performance nette de coûts
│   ├── factors.py             # Fama-French + stats perf + alpha FF4 + info ratio
│   ├── ml_regime/
│   │   ├── features.py        # feature engineering (CRSP+Compustat+FRED)
│   │   ├── labeling.py        # régimes HMM/GMM (euphoria / junk rally) + smoothing
│   │   ├── model.py           # walk-forward expanding window + seuil adaptatif
│   │   ├── evaluation.py      # métriques OOS + feature importance
│   │   └── signals.py         # SHORT_REDUCE / FULL_SHORT
│   ├── dynamic_short.py       # stratégie avec short dynamique piloté ML
│   └── plots.py               # toutes les figures → results/charts
├── notebooks/
│   └── value_strategy_narrative.ipynb   # notebook allégé qui importe les modules
├── results/charts/
└── tests/                     # tests légers sur la logique pure
```

## Principes de découpage

- **Une responsabilité par module**, testable et lisible isolément.
- **Logique métier extraite à l'identique** — les calculs ne changent pas. Le refactor élimine
  les variables globales (passage explicite de DataFrames entre fonctions) et **factorise la
  duplication IS/OOS** (les cellules 14-20 et 24 font le même pipeline de signaux deux fois →
  une seule fonction `build_signals(panel, start, end)`).
- **`config.py`** centralise tous les paramètres fixes du notebook (dates IS/OOS, seuils
  small/mid cap, `SECTOR_MIN_COUNT`, `REBAL_MONTHS`, `RF_ANNUAL`, seuils de sortie, poids short).

## Flux de données (`main.py`)

1. **data** : `wrds_loader` télécharge → `panel.build_panel()` merge → `intangibles` +
   `cleaning` → `cache.save_panel()` (parquet, écrase). `--use-cache` saute le fetch.
2. **signals** : `build_signals(panel, IS)` et `build_signals(panel, OOS)`.
3. **portfolio** : `construct_portfolios` + `compute_performance` → `perf_is`, `perf_oos`.
4. **factors** : FF4, tableaux alpha/stats.
5. **ml_regime** : features → labels → walk-forward → signaux euphoria.
6. **dynamic_short** : combine signal ML + perf OOS → `perf_h`.
7. **plots** : régénère toutes les figures dans `results/charts`.

## Contrainte d'exécution (honnêteté technique)

L'agent **n'a pas accès à WRDS** → le pipeline ne peut pas être exécuté de bout en bout ici.
Garanties : extraction **fidèle** de la logique (calculs identiques au notebook), imports
cohérents (`import value_strategy` passe), tests sur les fonctions pures sans WRDS (momentum,
ranking sectoriel, score qualité, smoothing de régimes, stats de performance sur séries
synthétiques). Le notebook narratif et `main.py` sont structurellement corrects mais à valider
par l'utilisateur avec un vrai accès WRDS.

## Tests légers (pytest)

- `test_signals.py` : momentum 6M skip-2, ranking sectoriel avec fallback global, score qualité
  composite sur petit DataFrame synthétique.
- `test_factors.py` : `performance_stats`, `alpha_ff4` sur séries connues.
- `test_ml_regime.py` : `smooth_regimes` (bridge + durée min), `find_optimal_threshold`.

## Hors scope (YAGNI)

Pas de CLI riche (argparse minimal suffit), pas de config YAML, pas de CI, pas de packaging PyPI,
pas de refactor des calculs eux-mêmes. Les notebooks de recherche d'origine (`Code_Value.ipynb`,
`Value_Strategy_GARP.ipynb`) sont déplacés dans `artefact/` (hors dépôt, ignoré par git), conservés
localement comme référence mais retirés du repo pour garder une arborescence propre.
