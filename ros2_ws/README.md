# ROS 2 Workspace

## Install

```bash
cd ~/isaaclab_ws/physplanet/ros2_ws
./install_ros2ws_deps.sh

conda activate ros2_env
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

## Launch

Parameters are managed through the config YAML files (`source/physplanet_assets/config/`);
the CLI can override them. Once the simulation is running, the current terrain information
is written automatically, so the mapping launch does not need a terrain argument.

The ROS 2 feature switches live in `src/physplanet_utils/config/ros2_config.yaml`:
toggle the elevation_mapping / depth_to_pointcloud / camera_info / local_height_scan /
friction_identifier / rviz nodes, and whether the ground-truth layers (friction_GT /
soil_params / terrain_class / rock_GT) are loaded into elevation mapping.

### Zhurong

```bash
# Terminal 1: simulation
conda activate env_isaaclab
source ros2_ws/install/setup.bash
source /opt/ros/ros_py311/install/setup.bash
~/IsaacLab/isaaclab.sh -p source/physplanet_assets/test/04_test_zhurong_mars_terrain.py

# Terminal 2: elevation mapping + RViz (after terminal 1)
conda activate ros2_env
source /opt/ros/humble/setup.bash
source ~/isaaclab_ws/physplanet/ros2_ws/install/setup.bash
ros2 launch physplanet_utils zhurong_elevation_mapping.launch.py
ros2 launch physplanet_utils zhurong_elevation_mapping.launch.py env_id:=1

# Terminal 3: keyboard control
conda activate ros2_env
source /opt/ros/humble/setup.bash
source ~/isaaclab_ws/physplanet/ros2_ws/install/setup.bash
python source/physplanet_assets/test/teleop_twist_keyboard.py
```

### Hunter

```bash
# Terminal 1: simulation
conda activate env_isaaclab
source ros2_ws/install/setup.bash
source /opt/ros/ros_py311/install/setup.bash
~/IsaacLab/isaaclab.sh -p source/physplanet_assets/test/03_test_hunter_mars_terrain.py

# Terminal 2: elevation mapping + RViz (after terminal 1)
conda activate ros2_env
source /opt/ros/humble/setup.bash
source ~/isaaclab_ws/physplanet/ros2_ws/install/setup.bash
ros2 launch physplanet_utils hunter_elevation_mapping.launch.py
ros2 launch physplanet_utils hunter_elevation_mapping.launch.py env_id:=1

# Terminal 3: keyboard control
conda activate ros2_env
source /opt/ros/humble/setup.bash
source ~/isaaclab_ws/physplanet/ros2_ws/install/setup.bash
python source/physplanet_assets/test/teleop_twist_keyboard.py
```
