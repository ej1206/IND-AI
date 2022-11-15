cmake_minimum_required(VERSION 3.0.2)
project(assign)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  roscpp
  rospy
  std_msgs
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  msg.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  srv.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ../../devel/include
)

## Declare a C++ executable
add_executable(${PROJECT_NAME} src/client.cpp)
add_executable(server src/server.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
)

target_link_libraries(server
  ${catkin_LIBRARIES}
)

add_dependencies (${PROJECT_NAME} assign)
add_dependencies (server assign)