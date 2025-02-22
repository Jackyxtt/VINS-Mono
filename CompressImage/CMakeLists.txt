cmake_minimum_required(VERSION 2.8.3)
project(compressed_image)

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

add_executable(compressed_image
    subscribeCompressedImage.cpp)


add_executable(infrared_image
    subscribeInfraredImage.cpp)

target_link_libraries(compressed_image ${catkin_LIBRARIES} ${OpenCV_LIBS})

target_link_libraries(infrared_image ${catkin_LIBRARIES} ${OpenCV_LIBS})
