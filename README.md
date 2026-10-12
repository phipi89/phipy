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
Inside this repository, the dedicated development project installs `phipy`
and sibling `../phipy-plots` editably:

```bash
uv sync --project dev
uv run --project dev python -m unittest discover -s tests -v
```

The local overrides live in `dev/pyproject.toml`, keeping them out of the
library's dependency metadata. Plain `uv sync` uses GitHub `phipy-plots`.

To use both local repositories editably from another project, add both
explicitly (adjust the paths for your layout):

First merge this override into that project's `pyproject.toml` so it can
replace the direct Git requirement:

```toml
[tool.uv]
override-dependencies = ["phipy-plots"]
```

```bash
uv add --editable ../phipy ../phipy-plots
```

Adding only local editable `phipy` uses the Git version of `phipy-plots`.
Add both paths explicitly when you want both repositories editable.
