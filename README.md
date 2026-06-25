# Value Strategy

Long/short equity strategy on US small/mid caps ($300M to $10B), rebalanced
semi-annually. The value signal is an intangible-adjusted book-to-market
(Peters & Taylor 2017), neutralized by sector, combined with a quality filter
and a momentum filter. A machine-learning module detects "junk rally" market
regimes to modulate the short exposure.

Data: CRSP and Compustat via WRDS, Fama-French factors, FRED macro.
In-sample 2003-2013, out-of-sample 2014-2024, with no recalibration between
the two.

The code starts from a research notebook of about 3,500 lines, here split into
a Python package. The pipeline runs with `main.py`; a narrative notebook
follows the same logic by importing the modules.

## The strategy

Value signal: `BM_adj = (book equity + KC + OC) / market cap`, where KC
capitalizes R&D and OC capitalizes SG&A. The idea is to correct the accounting
book value, which ignores intangible assets and therefore understates the value
of tech/pharma firms.

The signal is ranked within each GICS sector (and globally when a sector has
too few stocks), so as not to mechanically overweight structurally cheap
sectors (banks, energy, utilities).

On top of that come a composite quality score (ROCE, ROE, operating margin,
leverage) to avoid value traps, and a 6-month momentum (cumul t-6 to t-2,
skip-2) as an entry filter for the longs.

The long side keeps the top 20% sector B/M crossed with quality and positive
momentum; the short takes the bottom 25% quality within the growth bucket
(bottom 20% B/M). The backtest corrects survivorship bias (delisting returns,
Shumway 2001) and charges dynamic transaction costs (Amihud illiquidity) plus a
borrow fee that depends on market cap.

The ML part labels the regimes (HMM/GMM on liquidity and FRED macro features),
trains a classifier walk-forward (Logistic Regression and Gradient Boosting),
then cuts the short weight from 100% to 50% when the adverse regime is detected.

## Structure

```
main.py                       pipeline orchestrator
src/value_strategy/
    config.py                 parameters, paths, IS/OOS dates
    data/                     WRDS extraction, panel, intangibles, parquet cache
    signals.py                sector value, quality, momentum
    portfolio.py              long/short construction, net performance
    factors.py                Fama-French, stats, FF4 alpha
    ml_regime/                features, labeling, walk-forward, signals
    dynamic_short.py          ML-driven dynamic short
    plots.py                  figures
    reports.py                console summaries (IS vs OOS, verdict)
notebooks/
    value_strategy_narrative.ipynb
tests/
results/charts/
```

Each module has a single responsibility and is tested in isolation. The
computations are taken as-is from the notebook; the restructuring removes the
global variables (DataFrames are passed as arguments) and factors out the
signal pipeline, which was duplicated between IS and OOS.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`hmmlearn`, `xgboost` and the FRED clients are optional: without them, the code
falls back to `GaussianMixture`, `GradientBoosting` (scikit-learn) and a
synthetic macro panel.

Data access requires a WRDS account (CRSP + Compustat). Put your credentials in
a `.env.local` file (see `.env.local.example`); it is ignored by git:

```
WRDS_USERNAME=...
WRDS_PASSWORD=...
```

Otherwise, `wrds` prompts for them interactively on the first run.

## Usage

```bash
python main.py              # connects to WRDS, writes the cache, runs everything
python main.py --use-cache  # starts from the parquet cache, no WRDS
python main.py --no-plots   # without the figures
```

On each WRDS run, the panel and the ML inputs are re-downloaded then
overwritten in `data/cache/`. The panel is cached before the ML queries, so an
incident along the way does not lose the most expensive part. After that,
`--use-cache` replays the whole pipeline in about two minutes.

The `notebooks/value_strategy_narrative.ipynb` notebook does the same thing in
a commented form (set `USE_CACHE = True` after a first run). Figures go to
`results/charts/`.

## Results

Strategy net of costs, with no recalibration between IS and OOS.

| Metric | IS (2003-13) | OOS (2014-24) |
|---|---:|---:|
| Annual return | 12.0% | 16.0% |
| Sharpe | 0.92 | 1.18 |
| Max Drawdown | -12.0% | -12.8% |
| Annual FF4 alpha | 12.0% (t=3.5) | 15.7% (t=4.2) |
| Information Ratio vs HML | 1.09 | 1.34 |

The FF4 alpha stays significant (t > 2) over both periods, OOS included.

On the ML side, the walk-forward produces an out-of-sample AUC around 0.82
(Gradient Boosting) over 14 folds. The resulting dynamic short, compared to the
static version:

| | Static | Dynamic short |
|---|---:|---:|
| Sharpe (OOS) | 1.18 | 1.20 |
| CAGR (OOS) | 15.3% | 19.7% |
| Max Drawdown | -12.8% | -11.2% |
| $1 invested (IS+OOS) | $12.22 | $17.71 |

That is, a slightly higher Sharpe and a reduced drawdown.

Figures in `results/charts/`: cumulative wealth, 36-month rolling Sharpe,
annual returns, drawdowns, calendar heatmap, Sharpe by sub-period, ML regime
dashboard, static/dynamic short comparison and final summary. The console
prints the IS vs OOS table and the verdict on the dynamic short.

## Regime-detection robustness

The HMM/GMM regime clustering depends on the data vintage: on some extracts it
degenerates into a very imbalanced split (one regime above 90%), and the
walk-forward no longer has enough examples of the rare class to train.
`ml_regime.labeling` adds a guard: when the minority regime drops below
`MIN_REGIME_SHARE` (20%), it switches to a deterministic split by stress score
(mean of the standardized regime features, cutoff at the median). The active
regime remains the one with the highest VIX. Part 4 thus becomes reproducible
regardless of the extract.

## Quality score, a caveat

The composite quality score follows the original notebook convention: with
`ascending=False` on ROCE/ROE/OM and `ascending=True` on leverage, it is the
best fundamentals that get the lowest score, and the short selection
(`bottom 25%`) depends on it. The behavior is kept as-is so as not to change the
notebook's results. To flip the direction, just change the `ascending` flags in
`signals.QUALITY_SPECS`, but that changes all the performance figures.

## Tests

```bash
pytest
```

The tests cover the logic that does not depend on WRDS: skip-2 momentum, sector
rank and its global fallback, quality score, performance statistics, regime
smoothing and the anti-degeneracy guard, optimal threshold, borrow fees,
holding durations.

## References

- Peters & Taylor (2017), *Intangible capital and the investment-q relation*, JFE.
- Asness, Moskowitz & Pedersen (2013), *Value and Momentum Everywhere*, JF.
- Israel & Moskowitz (2013), *The role of shorting, firm size, and time on market anomalies*, JFE.
- Shumway (2001), *The Delisting Bias in CRSP Data*, JF.
- AQR (2019), *Quality Minus Junk*.

## License

MIT.
