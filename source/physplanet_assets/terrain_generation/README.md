# Terrain generation

Configurable terrain-world generator: DEM/HiRISE or procedural elevation sources,
Perlin terrain layers, rock-occupancy fields and per-region soil-parameter profiles.
All outputs are written as spatially aligned fields on a common grid
(`H`, `R`, `S`, `P`) together with the rendered/baked scene assets.

- `scripts/` — generation and baking pipeline (run inside Isaac Sim Python)
- `ui/webui.py` — browser UI for interactive generation
- `config/` — per-world configuration files (elevation source, layers, physics profiles)

Note: the large generated artifacts (`models/` — baked USD scene assets — and
`terrain/` — DEM rasters and intermediate rasters) are not stored in this repository.
Produce them with the scripts above using the shipped configs; outputs are written
to these two directories (git-ignored).

## Rock assets

The rock USD library (`models/rocks/`, ~0.6 GB for `mars_rocks`, used by scenes with
`rock_k > 0`) is not stored in this repository. Place a rock library under
`models/rocks/<name>` (see configs, e.g. `mars_rocks`) before generating scenes with
rocks, or set `rock_k: 0.0` for rock-free scenes. `textures/` and `heightmaps/` ARE
included, so Perlin scenes with texture quilting work out of the box.
