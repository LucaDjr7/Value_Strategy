#!/usr/bin/env python3
"""Full Value GARP strategy pipeline — stage-by-stage orchestrator.

Usage
-----
    python main.py                 # fetch WRDS, cache, run everything
    python main.py --use-cache     # start from the parquet cache (no WRDS)
    python main.py --no-plots      # skip figure generation

Stages: data -> signals (IS/OOS) -> backtest -> FF factors -> ML regime ->
dynamic short -> figures. All outputs go to ``results/``.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Allow `python main.py` without installation (adds src/ to the path)
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402

from value_strategy import config, dynamic_short, factors, plots, reports, signals  # noqa: E402
from value_strategy import portfolio as pf  # noqa: E402
from value_strategy.data import build as data_build  # noqa: E402
from value_strategy.ml_regime import pipeline as ml_pipeline  # noqa: E402


def run(use_cache: bool = False, make_plots: bool = True) -> None:
    config.ensure_dirs()

    # -- Part 1: data (panel cached before the ML fetch) --
    panel, df_comp, crsp_ml, macro = data_build.acquire_data(use_cache=use_cache)

    # -- Part 2: signals + IS backtest --
    print("\n" + "=" * 60 + "\n  PART 2 — IN-SAMPLE (2003-2013)\n" + "=" * 60)
    panel_sm_is, filtered_is = signals.build_signals(panel, config.IS_START, config.IS_END)
    long_is, short_is, _ = pf.construct_portfolios(filtered_is, config.IS_START, config.IS_END)
    perf_is, costs_is = pf.compute_performance(
        panel_sm_is, long_is, short_is, config.IS_START, config.IS_END,
    )

    # -- Part 3: signals + OOS backtest + factors --
    print("\n" + "=" * 60 + "\n  PART 3 — OUT-OF-SAMPLE (2014-2024)\n" + "=" * 60)
    panel_sm_oos, filtered_oos = signals.build_signals(panel, config.OOS_START, config.OOS_END)
    long_oos, short_oos, _ = pf.construct_portfolios(filtered_oos, config.OOS_START, config.OOS_END)
    perf_oos, costs_oos = pf.compute_performance(
        panel_sm_oos, long_oos, short_oos, config.OOS_START, config.OOS_END,
    )

    ff5, mom_ff = factors.load_ff_factors()
    stats_is, stats_oos = reports.print_is_oos_comparison(
        perf_is, perf_oos, ff5, mom_ff, costs_is, costs_oos,
    )

    # -- Part 4: ML regime detection --
    ml = ml_pipeline.run_ml_regime(crsp_ml, df_comp, macro)

    # -- Part 5: dynamic short --
    ff = factors.download_ff_factors()
    perf_h, summary = dynamic_short.build_dynamic_short(perf_oos, ml["signals_df"], costs_oos)
    perf_h, metrics = dynamic_short.attach_ff_and_metrics(perf_h, ff)
    ff4_results = dynamic_short.ff4_regression(perf_h, ff)
    reports.print_dynamic_short_verdict(perf_h, metrics, ff4_results, summary, costs_oos)

    # -- Diagnostic: holding durations --
    print("\nAverage holding durations:")
    for label, snaps in [("LONG IS", long_is), ("SHORT IS", short_is),
                         ("LONG OOS", long_oos), ("SHORT OOS", short_oos)]:
        avg, med, n = pf.holding_durations(snaps)
        print(f"  {label:>10s}: mean={avg:.1f} months  median={med:.1f} months  (n={n})")

    # -- Figures --
    if make_plots:
        print("\n" + "=" * 60 + "\n  FIGURE GENERATION\n" + "=" * 60)
        plots.setup_style()
        plots.plot_cumulative_wealth(perf_is, perf_oos, ff5)
        plots.plot_rolling_sharpe(perf_is, perf_oos, ff5)
        plots.plot_annual_returns(perf_is, perf_oos, ff5)
        plots.plot_drawdown(perf_is, perf_oos, ff5)
        plots.plot_calendar_heatmap(perf_oos)
        plots.plot_robustness(perf_is, perf_oos, ff5)
        plots.plot_ml_dashboard(ml["mkt_df"], ml["results_df"], ml["importance_df"],
                                ml["metrics_df"], ml["regime_method"], ml["avg_threshold"])
        plots.plot_dynamic_short(perf_h, metrics, ff4_results, summary)
        plots.plot_final_comparison(perf_is, perf_oos, perf_h, ff5,
                                    stats_is, stats_oos, metrics["adj"])

    print("\nPipeline finished. Results in results/.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Value GARP strategy pipeline")
    parser.add_argument("--use-cache", action="store_true",
                        help="start from the parquet cache without a WRDS connection")
    parser.add_argument("--no-plots", action="store_true",
                        help="do not generate the figures")
    args = parser.parse_args()
    run(use_cache=args.use_cache, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
