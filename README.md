# Value Strategy

Stratégie long/short equity sur les small/mid caps américaines (300 M$ à 10 Md$),
rebalancement semestriel. Le signal de valeur est un book-to-market ajusté du capital
immatériel (Peters & Taylor 2017), neutralisé par secteur, croisé avec un filtre de qualité
et un filtre de momentum. Un module de machine learning détecte les régimes de marché de type
"junk rally" pour moduler l'exposition short.

Données : CRSP et Compustat via WRDS, facteurs Fama-French, macro FRED.
In-sample 2003-2013, out-of-sample 2014-2024, sans recalibrage entre les deux.

Le code part d'un notebook de recherche d'environ 3 500 lignes, ici découpé en un package
Python. Le pipeline se relance avec `main.py` ; un notebook narratif reprend la même logique en
important les modules.

## La stratégie

Signal value : `BM_adj = (capitaux propres + KC + OC) / market cap`, où KC capitalise la R&D et
OC le SG&A. L'idée est de corriger le book value comptable, qui ignore les actifs immatériels et
sous-estime donc la valeur des entreprises tech/pharma.

Le signal est ranké à l'intérieur de chaque secteur GICS (et globalement quand un secteur compte
trop peu de titres), pour ne pas surpondérer mécaniquement les secteurs structurellement cheap
(banques, énergie, utilities).

À ça s'ajoutent un score de qualité composite (ROCE, ROE, marge opérationnelle, levier) pour
éviter les value traps, et un momentum 6 mois (cumul t-6 à t-2, skip-2) en filtre d'entrée des
longs.

Le long retient le top 20 % B/M sectoriel croisé qualité et momentum positif ; le short prend le
bottom 25 % de qualité dans le bucket growth (bottom 20 % B/M). Le backtest corrige le
survivorship bias (rendements de delisting, Shumway 2001) et facture des coûts de transaction
dynamiques (illiquidité d'Amihud) plus un borrow fee fonction de la capitalisation.

La partie ML labélise les régimes (HMM/GMM sur features de liquidité et macro FRED), entraîne un
classifieur en walk-forward (Logistic Regression et Gradient Boosting), puis réduit le poids du
short de 100 % à 50 % quand le régime adverse est détecté.

## Structure

```
main.py                       orchestrateur du pipeline
src/value_strategy/
    config.py                 paramètres, chemins, dates IS/OOS
    data/                     extraction WRDS, panel, intangibles, cache parquet
    signals.py                value sectoriel, qualité, momentum
    portfolio.py              construction long/short, performance nette
    factors.py                Fama-French, stats, alpha FF4
    ml_regime/                features, labeling, walk-forward, signaux
    dynamic_short.py          short dynamique piloté par le ML
    plots.py                  figures
    reports.py                synthèses console (IS vs OOS, verdict)
notebooks/
    value_strategy_narrative.ipynb
tests/
results/charts/
```

Chaque module a une responsabilité unique et se teste isolément. Les calculs sont repris tels
quels depuis le notebook ; la restructuration enlève les variables globales (les DataFrames
circulent en arguments) et factorise le pipeline de signaux, qui était dupliqué entre IS et OOS.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`hmmlearn`, `xgboost` et les clients FRED sont optionnels : sans eux, le code retombe sur
`GaussianMixture`, `GradientBoosting` (scikit-learn) et un panel macro synthétique.

L'accès aux données suppose un compte WRDS (CRSP + Compustat). Renseignez les identifiants dans
un fichier `.env.local` (voir `.env.local.example`), il est ignoré par git :

```
WRDS_USERNAME=...
WRDS_PASSWORD=...
```

À défaut, `wrds` les demande en interactif au premier lancement.

## Utilisation

```bash
python main.py              # se connecte à WRDS, écrit le cache, exécute tout
python main.py --use-cache  # repart du cache parquet, sans WRDS
python main.py --no-plots   # sans les figures
```

À chaque run avec WRDS, le panel et les entrées du ML sont retéléchargés puis écrasés dans
`data/cache/`. Le panel est mis en cache avant les requêtes ML, donc un incident en cours de
route ne fait pas perdre la partie la plus coûteuse. Ensuite `--use-cache` rejoue tout le
pipeline en deux minutes environ.

Le notebook `notebooks/value_strategy_narrative.ipynb` fait la même chose de façon commentée
(mettre `USE_CACHE = True` après un premier run). Les figures vont dans `results/charts/`.

## Résultats

Stratégie nette de coûts, sans recalibrage entre IS et OOS.

| Métrique | IS (2003-13) | OOS (2014-24) |
|---|---:|---:|
| Rendement annuel | 12.0 % | 16.0 % |
| Sharpe | 0.92 | 1.18 |
| Max Drawdown | -12.0 % | -12.8 % |
| Alpha FF4 annuel | 12.0 % (t=3.5) | 15.7 % (t=4.2) |
| Information Ratio vs HML | 1.09 | 1.34 |

L'alpha FF4 reste significatif (t > 2) sur les deux périodes, OOS compris.

Côté ML, le walk-forward sort une AUC out-of-sample autour de 0.82 (Gradient Boosting) sur
14 folds. Le short dynamique qui en découle, comparé à la version statique :

| | Statique | Short dynamique |
|---|---:|---:|
| Sharpe (OOS) | 1.18 | 1.20 |
| CAGR (OOS) | 15.3 % | 19.7 % |
| Max Drawdown | -12.8 % | -11.2 % |
| 1 $ investi (IS+OOS) | 12.22 $ | 17.71 $ |

Soit un Sharpe en légère hausse et un drawdown réduit.

Figures dans `results/charts/` : richesse cumulée, rolling Sharpe 36 mois, rendements annuels,
drawdowns, calendar heatmap, Sharpe par sous-période, dashboard ML des régimes, comparaison short
statique/dynamique et synthèse finale. La console imprime le tableau IS vs OOS et le verdict sur
le short dynamique.

## Robustesse de la détection de régime

Le clustering HMM/GMM des régimes dépend du millésime des données : sur certaines extractions il
dégénère en un partage très déséquilibré (un régime au-delà de 90 %), et le walk-forward n'a plus
assez d'exemples de la classe rare pour s'entraîner. `ml_regime.labeling` ajoute un garde-fou :
quand le régime minoritaire passe sous `MIN_REGIME_SHARE` (20 %), on bascule sur un découpage
déterministe par score de stress (moyenne des features de régime standardisées, coupure à la
médiane). Le régime actif reste celui au VIX le plus élevé. La partie 4 redevient ainsi
reproductible quelle que soit l'extraction.

## Score de qualité, à savoir

Le score de qualité composite reprend la convention du notebook d'origine : avec `ascending=False`
sur ROCE/ROE/OM et `ascending=True` sur le levier, ce sont les meilleurs fondamentaux qui
obtiennent le score le plus bas, et la sélection du short (`bottom 25 %`) en dépend. Le
comportement est conservé tel quel pour ne pas changer les résultats du notebook. Pour inverser la
direction, il suffit de changer les `ascending` dans `signals.QUALITY_SPECS`, mais cela modifie
toutes les performances.

## Tests

```bash
pytest
```

Les tests portent sur la logique qui ne dépend pas de WRDS : momentum skip-2, rank sectoriel et
son fallback global, score de qualité, statistiques de performance, lissage des régimes et
garde-fou anti-dégénérescence, seuil optimal, borrow fees, durées de détention.

## Références

- Peters & Taylor (2017), *Intangible capital and the investment-q relation*, JFE.
- Asness, Moskowitz & Pedersen (2013), *Value and Momentum Everywhere*, JF.
- Israel & Moskowitz (2013), *The role of shorting, firm size, and time on market anomalies*, JFE.
- Shumway (2001), *The Delisting Bias in CRSP Data*, JF.
- AQR (2019), *Quality Minus Junk*.

## Licence

MIT.
