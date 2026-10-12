# phipy

A personal collection of Python utilities that I keep reusing across projects.

`phipy` collects pragmatic helpers for everyday scientific work: arrays, signals, geometry, plotting, meshes, and notebook setup. This is a convenience layer; many functions are small wrappers around existing tools, adapted to my use cases or just there as a bookmark.

No plans for backwards compatibility; intended to be vendored into projects when archived.

### Honorable mentions:
```
# setup autoreload, and populate the namespace with tqdm, plt, np, etc.
from phipy.full import *

# moving window average
support, yvals = pp.signal.moving_window(x, y, window=5)

# mesh processing
smoothed = pp.mesh.remesh_in_instant_meshes(mesh)
```

### Setup
Requires Python 3.13 or newer. Installing from GitHub also installs
`phipy-plots` from its GitHub `v0.2.0` tag; no local checkouts are needed.

```bash
uv add git+https://github.com/phipi89/phipy
```

### Local development
Inside this repository, `uv sync` installs `phipy` and sibling
`../phipy-plots` editably through the local `[tool.uv.sources]` override.

To use both local repositories editably from another project, add both
explicitly (adjust the paths for your layout):

```bash
uv add --editable ../phipy ../phipy-plots
```

Dependency projects' uv source overrides are not inherited by the consuming
project. Adding only local editable `phipy` therefore still uses the Git
version of `phipy-plots`.
