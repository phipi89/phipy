"""Full notebook workspace import.

Use::

    from phipy.full import *

to get numpy, pathlib, tqdm, and all phipy-plots plotting functions in one
import. Autoreload and matplotlib backend are configured automatically
when running inside an IPython/jupyter notebook.
This also flattens all relevant submodules directly into the global namespace,
e.g., `normalized`, `identity`, `closest_point`, `red`, etc.
"""

import pathlib

import numpy as np
import scipy
from tqdm.auto import tqdm

import phipy as pp
from phipy.notebook import setup as _setup

from .array import *
from .geometry import *
from .mesh import *
from .plots import *
from .signal import *
from .transform import *
from .utils import *

_setup()
