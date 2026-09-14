# physplanet_assets

Rover assets, the configurable terrain-field generator and simulation test entry points.

```text
physplanet_assets/     # Zhurong / Hunter assets (URDF + USD), wheel configurations
terrain_generation/
  config/              # terrain parameter configs
  scripts/             # elevation, rock, physics-layer and USD generation
  ui/                  # terrain generation WebUI
  models/              # generation inputs: textures/ and heightmaps/ (rocks/ excluded)
test/                  # hard/soft terrain tests, sim config WebUI, teleoperation
utils/                 # Hunter / Zhurong control and sensors, terramechanics.py, sim_utils.py
tools/                 # URDF generation and material binding
config/                # per-rover simulation configs (hunter_config.yaml, zhurong_config.yaml)
```

## Terrain generation (command line)

Install the conversion dependencies once:

```bash
~/IsaacLab/_isaac_sim/python.sh -m pip install pycollada open3d
```

Configs set `terrain_name` (the output directory). Generate with:

```bash
# Default mixed terrain
~/IsaacLab/isaaclab.sh -p terrain_generation/generate_terrain.py --config terrain_generation/config/terrain_config.yaml

# Layered-soil test terrains
~/IsaacLab/isaaclab.sh -p terrain_generation/generate_terrain.py --config terrain_generation/config/terrain_flat_config.yaml
~/IsaacLab/isaaclab.sh -p terrain_generation/generate_terrain.py --config terrain_generation/config/terrain_slope_config.yaml

# Lunar visual terrain (reuses existing rock geometry, binds lunar materials)
~/IsaacLab/isaaclab.sh -p terrain_generation/generate_terrain.py --config terrain_generation/config/lunar_terrain_config.yaml

# Height-scan navigation scenes used by the RL baseline
bash ../../physplanet_tasks/scripts/generate_heightscan_easy.sh   # also: medium / hard
```

(Run from `source/physplanet_assets/`.)

### Terrain generation GUI

```bash
~/IsaacLab/_isaac_sim/python.sh -m pip install --upgrade \
  "gradio==4.44.1" "gradio-client==1.3.0" "fastapi==0.115.7" \
  "starlette==0.45.3" "pydantic==2.11.5" "websockets==11.0.3"

~/IsaacLab/isaaclab.sh -p terrain_generation/ui/webui.py
# open http://localhost:7860
```

Note: the rock USD library (`models/rocks/`) is not distributed with the repository.
Scenes with `rock_k > 0` require it; either provide the library or set `rock_k: 0.0`.
`textures/` and `heightmaps/` are included, so Perlin scenes work out of the box.

## Simulation tests

Config files: `config/hunter_config.yaml`, `config/zhurong_config.yaml`.

### Hunter

```bash
conda activate env_isaaclab
source ros2_ws/install/setup.bash
source /opt/ros/ros_py311/install/setup.bash

# Hard terrain
~/IsaacLab/isaaclab.sh -p test/03_test_hunter_mars_terrain.py
# Soft terrain (optionally select a terrain)
~/IsaacLab/isaaclab.sh -p test/05_test_hunter_terramechanics.py --num_envs 1
~/IsaacLab/isaaclab.sh -p test/05_test_hunter_terramechanics.py --terrain terrain_flat --num_envs 1
```

### Zhurong

```bash
conda activate env_isaaclab
source ros2_ws/install/setup.bash
source /opt/ros/ros_py311/install/setup.bash

# Hard terrain
~/IsaacLab/isaaclab.sh -p test/04_test_zhurong_mars_terrain.py
~/IsaacLab/isaaclab.sh -p test/04_test_zhurong_mars_terrain.py --terrain terrain_2 --num_envs 1
# Soft terrain
~/IsaacLab/isaaclab.sh -p test/06_test_zhurong_terramechanics.py --num_envs 1
~/IsaacLab/isaaclab.sh -p test/06_test_zhurong_terramechanics.py --terrain terrain_slope --num_envs 1
```

### Simulation configuration GUI

```bash
~/IsaacLab/isaaclab.sh -p test/sim_webui.py
# open http://localhost:7861
```

### Keyboard teleoperation (ROS 2)

```bash
conda activate ros2_env
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
python3 test/teleop_twist_keyboard.py
```

## Zhurong USD generation

```bash
conda activate env_isaaclab

# Step 1: generate the URDF (package:// paths; syncs into the ROS description package)
python3 tools/generate_zhurong_urdf.py

# Step 2: URDF -> USD
export ROS_PACKAGE_PATH=$(pwd)/../../ros2_ws/src${ROS_PACKAGE_PATH:+:${ROS_PACKAGE_PATH}}
~/IsaacLab/isaaclab.sh -p ~/IsaacLab/scripts/tools/convert_urdf.py \
  physplanet_assets/zhurong_assets/zhurong.urdf \
  physplanet_assets/zhurong_assets/zhurong.usd

# Step 3: bind materials
python3 tools/bind_zhurong_materials.py
```

(Run from `source/physplanet_assets/`.)
