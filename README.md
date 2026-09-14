# PhysPlanet

![cover](docs/pictures/covezhurong_mars.png)

A physics-aware simulation and learning platform for planetary rovers, built on Isaac Lab.
Terrain elevation, rock occupancy, semantics and soil physical properties are represented as
**spatially aligned fields**; the same fields drive rendering, collision, the vectorized
semi-empirical wheel–terrain model, simulator ground truth and policy observations. The
platform supports height-scan navigation for the Hunter SE off-road rover and the Zhurong
Mars rover, and ships a geometry / +semantic / +physics observation-ablation baseline.

> This project is under active development; interfaces and features may change.

## Requirements

- Ubuntu 22.04
- NVIDIA RTX GPU (32 GB VRAM recommended for large batch sizes)
- Isaac Sim 5.0
- Isaac Lab 2.2 (the `isaaclab` Python package)
- conda environment `env_isaaclab` (Isaac Sim, Isaac Lab and PyTorch)
- Optional: ROS 2 Humble (online elevation/friction mapping in `ros2_ws/`, conda env `ros2_env`)

## Installation

Install Isaac Lab following the official guide:
[isaac-sim.github.io/IsaacLab/v2.2.1](https://isaac-sim.github.io/IsaacLab/v2.2.1/source/setup/installation/binaries_installation.html)

```bash
pip install -e source/physplanet_assets
pip install -e source/physplanet_tasks
pip install -e source/physplanet_rl
```

## Navigation baseline (observation ablation)

`source/physplanet_tasks` registers three observation conditions (Zhurong rover,
soft-soil terrain):

| Task ID | Observation |
|---------|-------------|
| `Isaac-ZhurongNavHeightscan-v10` | **Baseline**: height scan + dimension-matched zero-filled semantic/physics fields |
| `Isaac-ZhurongNavHeightscan-v11` | **+Semantic**: aligned one-hot semantic field |
| `Isaac-ZhurongNavHeightscan-v12` | **+Physics**: semantic + aligned physical field (n0, φ) |
| `Isaac-HunterNavHeightscan-v0` | Hunter SE chassis baseline |

Train and play:

```bash
python source/physplanet_rl/physplanet_rl/train.py --task Isaac-ZhurongNavHeightscan-v10 --num_envs 512
python source/physplanet_rl/physplanet_rl/play.py  --task Isaac-ZhurongNavHeightscan-v10
```

The terrain scene is selected through the `ROVER_TERRAIN` environment variable
(default `rl_demo/heightscan/01_easy`).

## Quick start (GUI)

Terrain generation WebUI:

```bash
~/IsaacLab/isaaclab.sh -p source/physplanet_assets/terrain_generation/ui/webui.py
# open http://localhost:7860
```

Simulation configuration WebUI:

```bash
~/IsaacLab/isaaclab.sh -p source/physplanet_assets/test/sim_webui.py
# open http://localhost:7861
```

Command-line terrain generation, hard/soft terrain tests and ROS 2 teleoperation are
documented in [source/physplanet_assets/README.md](source/physplanet_assets/README.md);
ROS 2 elevation mapping is documented in [ros2_ws/README.md](ros2_ws/README.md).

## License

Released under the [MIT License](LICENSE). Portions adapted from
[RLRoverLab](https://github.com/abmoRobotics/RLRoverLab), WheeledLab and Isaac Lab retain
their original license headers — see [THIRD_PARTY.md](THIRD_PARTY.md).

## Citation

The paper describing PhysPlanet is under review; citation information will be added upon
publication.
