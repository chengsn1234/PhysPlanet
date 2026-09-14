cmake_minimum_required(VERSION 3.8)
project(physplanet_utils)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# Install Python scripts
install(PROGRAMS
  scripts/terrain_marker_publisher.py
  scripts/zhurong_camera_info_publisher.py
  scripts/zhurong_depth_to_pointcloud.py
  scripts/zhurong_obs_camera_info_publisher.py
  scripts/zhurong_obs_depth_to_pointcloud.py
  scripts/hunter_camera_info_publisher.py
  scripts/hunter_depth_to_pointcloud.py
  scripts/hunter_elevation_to_heightscan.py
  scripts/hunter_local_height_scan.py
  scripts/zhurong_elevation_to_heightscan.py
  scripts/zhurong_local_height_scan.py
  scripts/friction_identifier.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()
