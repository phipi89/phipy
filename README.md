# phipy

A personal collection of Python utilities that I keep reusing across projects.

`phipy` collects pragmatic helpers for everyday scientific work: arrays, signals, geometry, images, meshes, and notebook setup. This is a convenience layer; many functions are small wrappers around existing tools, adapted to my use cases or just there as a bookmark.

No plans for backwards compatibility; intended to be vendored into projects when archived.

Honorable mentions:
```
# setup autoreload, and populate the namespace with tqdm, plt, np, etc.
from phipy.full import *

# moving window average
support, yvals = pp.signal.moving_window(x, y, window=5)

# mesh processing
smoothed = pp.mesh.remesh_in_instant_meshes(mesh)
```
