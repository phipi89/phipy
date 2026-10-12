import phipy_plots as _phipy_plots
from phipy_plots import *

__path__ = _phipy_plots.__path__

if hasattr(_phipy_plots, "__all__"):
    __all__ = _phipy_plots.__all__


def __getattr__(name):
    return getattr(_phipy_plots, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_phipy_plots)))
