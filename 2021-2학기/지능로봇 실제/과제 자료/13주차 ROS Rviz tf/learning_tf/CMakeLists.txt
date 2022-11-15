cmake_minimum_required(VERSION 3.0.2)
project(learning_tf)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  tf
  turtlesim
)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)


add_executable(turtle_tf_broadcaster src/turtle_broadcaster.cpp)

target_link_libraries(turtle_tf_broadcaster ${catkin_LIBRARIES})