cmake_minimum_required(VERSION 2.8.3)
project(feature_tracker)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    sensor_msgs
    cv_bridge
    camera_model
    )

find_package(OpenCV REQUIRED)

catkin_package()

include_directories(
    ${catkin_INCLUDE_DIRS}
    )

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(feature_tracker
    src/feature_tracker_node.cpp
    src/parameters.cpp
    src/feature_tracker.cpp
    )

target_link_libraries(feature_tracker ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(test_pyrLK src/test_pyrLK.cpp)
target_link_libraries(test_pyrLK ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(test_pyrLK1 src/test_pyrLK1.cpp)
target_link_libraries(test_pyrLK1 ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(test_buildPyr src/test_buildPyr.cpp)
target_link_libraries(test_buildPyr ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(test_pyrLKflow src/test_pyrLKflow.cpp)
target_link_libraries(test_pyrLKflow ${catkin_LIBRARIES} ${OpenCV_LIBS})