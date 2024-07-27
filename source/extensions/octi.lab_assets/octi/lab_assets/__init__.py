# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing asset and sensor configurations."""

import os
import toml

# Conveniences to other module directories via relative paths
OCTILAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
"""Path to the extension source directory."""
OCTILAB_ASSETS_DATA_DIR = os.path.join(OCTILAB_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""

OCTILAB_ASSETS_METADATA = toml.load(os.path.join(OCTILAB_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = OCTILAB_ASSETS_METADATA["package"]["version"]
