cmake_minimum_required(VERSION 3.0.2)
project(pcl)


find_package(catkin REQUIRED COMPONENTS
  message_generation
  pcl_conversions
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  std_msgs
)
find_package(PCL_REQUIRED)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)


add_executable(${PROJECT_NAME} src/pcl.cpp)


target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
)

