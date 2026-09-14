#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
PYTHON=${PYTHON:-python}

source "$SCRIPT_DIR/isaac_usd_env.sh"
setup_isaac_usd_env "$PYTHON"

"$PYTHON" \
  "$ROOT_DIR/source/physplanet_assets/terrain_generation/generate_terrain.py" \
  --config "$ROOT_DIR/source/physplanet_assets/terrain_generation/config/rl_demo/heightscan/01_easy.yaml"
