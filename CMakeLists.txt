cmake_minimum_required(VERSION 3.8)
project(hdc_robot_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# C++17 required for HDC operations
set(CMAKE_CXX_STANDARD 17)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(pcl_ros REQUIRED)

# Optional CUDA support
find_package(CUDA QUIET)
if(CUDA_FOUND)
  enable_language(CUDA)
  add_compile_definitions(WITH_CUDA)
endif()

# Include directories
include_directories(
  include
  ${CMAKE_CURRENT_SOURCE_DIR}/hdc_core/include
)

# HDC Core C++ Library
add_library(hdc_core SHARED
  hdc_core/src/hypervector.cpp
  hdc_core/src/operations.cpp
  hdc_core/src/memory.cpp
  hdc_core/src/encoding/spatial_encoder.cpp
  hdc_core/src/encoding/visual_encoder.cpp
  hdc_core/src/encoding/temporal_encoder.cpp
  hdc_core/src/reasoning/associative_memory.cpp
  hdc_core/src/reasoning/similarity.cpp
)

target_include_directories(hdc_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/hdc_core/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(hdc_core
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
)

# CUDA kernels if available
if(CUDA_FOUND)
  add_library(hdc_cuda SHARED
    hdc_core/src/cuda/bundle_kernel.cu
    hdc_core/src/cuda/bind_kernel.cu
    hdc_core/src/cuda/similarity_kernel.cu
  )
  target_link_libraries(hdc_core hdc_cuda)
  set_property(TARGET hdc_cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

# ROS 2 Nodes
add_executable(perception_node
  ros2_nodes/perception_node/src/perception_node.cpp
  ros2_nodes/perception_node/src/multimodal_encoder.cpp
)

ament_target_dependencies(perception_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  image_transport
  pcl_ros
)

target_link_libraries(perception_node hdc_core)

add_executable(control_node
  ros2_nodes/control_node/src/control_node.cpp
  ros2_nodes/control_node/src/fault_tolerant_controller.cpp
)

ament_target_dependencies(control_node
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
)

target_link_libraries(control_node hdc_core)

add_executable(learning_node
  ros2_nodes/learning_node/src/learning_node.cpp
  ros2_nodes/learning_node/src/behavior_learner.cpp
)

ament_target_dependencies(learning_node
  rclcpp
  std_msgs
)

target_link_libraries(learning_node hdc_core)

# Install targets
install(TARGETS
  hdc_core
  perception_node
  control_node
  learning_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install headers
install(DIRECTORY hdc_core/include/
  DESTINATION include
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python scripts
install(PROGRAMS
  scripts/demo_controller.py
  scripts/benchmark_suite.py
  DESTINATION lib/${PROJECT_NAME}
)

# Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  
  ament_lint_auto_find_test_dependencies()
  
  # C++ tests
  ament_add_gtest(test_hypervector
    test/test_hypervector.cpp
  )
  target_link_libraries(test_hypervector hdc_core)
  
  ament_add_gtest(test_operations
    test/test_operations.cpp
  )
  target_link_libraries(test_operations hdc_core)
  
  # Python tests
  ament_add_pytest_test(test_controller test/test_controller.py)
endif()

ament_export_dependencies(
  rclcpp
  rclpy
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  cv_bridge
  image_transport
  pcl_ros
)

ament_export_include_directories(include)
ament_export_libraries(hdc_core)

ament_package()