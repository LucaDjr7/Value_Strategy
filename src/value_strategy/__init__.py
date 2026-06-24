"""Stratégie long/short equity value, qualité, momentum, avec détection ML.

Le pipeline suit cinq parties (détaillées dans le README) : la couche data
(extraction WRDS, panel mensuel, intangibles, nettoyage), les signaux (value
sectoriel, qualité, momentum), le portefeuille (construction long/short et
performance nette), la détection de régime (ml_regime) et le short dynamique.

Les sous-modules qui tirent des dépendances lourdes (WRDS, matplotlib) sont
importés à la demande, pour qu'un simple `import value_strategy` reste léger.
"""

from . import config

__all__ = ["config"]
__version__ = "1.0.0"
