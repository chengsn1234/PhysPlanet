# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import toml

# wheeledlab_assets
# Conveniences to other module directories via relative paths
WHEELEDLAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
"""Path to the extension source directory."""
WHEELEDLAB_ASSETS_DATA_DIR = os.path.join(WHEELEDLAB_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""
WHEELEDLAB_ASSETS_METADATA = toml.load(os.path.join(WHEELEDLAB_ASSETS_EXT_DIR, "config", "extension.toml"))
# Configure the module-level variables
__version__ = WHEELEDLAB_ASSETS_METADATA["package"]["version"]

HUNTER_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "hunter_assets"))
ZHURONG_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "zhurong_assets"))

# Import the assets
from .hunter import *
from .zhurong import *
