"""Long/short equity value, quality, momentum strategy, with ML detection.

The pipeline follows five parts (detailed in the README): the data layer (WRDS
extraction, monthly panel, intangibles, cleaning), the signals (sector value,
quality, momentum), the portfolio (long/short construction and net
performance), the regime detection (ml_regime) and the dynamic short.

Sub-modules that pull heavy dependencies (WRDS, matplotlib) are imported on
demand, so that a plain `import value_strategy` stays lightweight.
"""

from . import config

__all__ = ["config"]
__version__ = "1.0.0"
