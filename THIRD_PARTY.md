# Third-Party Notices

Original PhysPlanet contributions are released under the MIT License. Files that
retain another SPDX header, and third-party software, data, or assets, remain
under their respective licenses and terms.

| Component | Use in PhysPlanet | Notes |
| --- | --- | --- |
| Isaac Sim / Isaac Lab | Simulation runtime and extension framework | Install and use under NVIDIA's applicable terms. |
| Isaac Lab extension templates | Package layout and selected training entry-point structure | Files retaining the `BSD-3-Clause` header remain under the Isaac Lab license; see [`LICENSES/BSD-3-Clause-IsaacLab.txt`](LICENSES/BSD-3-Clause-IsaacLab.txt). |
| [RLRoverLab](https://github.com/abmoRobotics/RLRoverLab) | Reference implementation for selected navigation observations, rewards, and terminations | Adapted files retain an `Apache-2.0` header; see [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt). |
| [MarsSim](https://github.com/WMR-team/MarsSim) | Source lineage for the vectorized terramechanics model and rock-distribution implementation | The upstream repository does not currently declare a repository-wide source-code license. Confirm and record redistribution permission for code-derived portions before a public release; cite [Zhou et al. (2023)](https://doi.org/10.1109/TAES.2022.3207705). |
| [skrl](https://github.com/Toni-SM/skrl) / [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) | Reinforcement-learning algorithms and environment API | Installed separately; see the upstream project licenses. |
| ROS 2 Humble | Robot middleware | See the ROS 2 distribution licenses. |
| ROS 2 Python 3.11 compatibility runtime | Bundled as `ros_py311/ros_py311.tar.xz` for Isaac Sim compatibility | Derived from [camopel/isaacsim-ros2-python3.11](https://github.com/camopel/isaacsim-ros2-python3.11), licensed under Apache-2.0. See `ros_py311/source.md` and the notices inside the archive. |
| [elevation_mapping_cupy](https://github.com/leggedrobotics/elevation_mapping_cupy) | Elevation mapping integration | See the upstream project and files under `ros2_ws/src/elevation_mapping_cupy`. |
| [HiRISE / PDS](https://www.uahirise.org/) | Optional Mars DTM and ortho-image downloads | Data is downloaded by the terrain UI and is not redistributed in this repository. Cite the original product and follow PDS/HiRISE terms. |
| Open3D, trimesh, pycollada, Gradio | Terrain generation and UI dependencies | Installed separately; see each upstream package license. |

The bundled rover and rock assets must be used subject to the provenance and
license information shipped with their original sources. Do not assume that
the MIT License applies to third-party meshes, textures, USD files, or data.

| Vendored ROS 2 packages (`ros2_ws/src/`) | Vendored source | `ros2_numpy` and a fork of `elevation_mapping_cupy` are vendored under their original licenses; see headers in each package. |
| [WheeledLab](https://github.com/isaac-sim/WheeledLab) | Asset configuration structure and asset-directory conventions in `physplanet_assets` | Internal module-level names retained; see file headers |
