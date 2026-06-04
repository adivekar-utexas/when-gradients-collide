"""Algorithm implementations: OPRO, GPO, TextGrad, PE2.

Each module contains the algorithm-specific Optimizer, GradientComputer,
and LossComputer subclasses co-located with the algorithm class for cohesion.
"""

from .gpo import GPO
from .opro import OPRO
from .pe2 import PE2
from .textgrad import TextGrad

__all__ = [
    "OPRO",
    "GPO",
    "PE2",
    "TextGrad",
]
