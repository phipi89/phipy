"""Full notebook workspace import.

Use::

    from phipy.full import *

to get numpy, pathlib, tqdm, and all figtools plotting functions in one
import. Autoreload and matplotlib backend are configured automatically
when running inside an IPython/jupyter notebook.
"""

import pathlib

import numpy as np
from figtools.matplotlib import *
from figtools.plotly import *
from tqdm.auto import tqdm

import phipy as pp
from phipy.notebook import setup as _setup

_setup()
