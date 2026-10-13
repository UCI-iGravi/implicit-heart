"""Shape interpolation of cardiac masks with variational implicit functions (Turk & O'Brien, 1999)."""
from .data import Data
from .implicit import ImplicitFunction

__all__ = ["Data", "ImplicitFunction"]
