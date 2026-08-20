cmake_minimum_required(VERSION 3.10)
project(PanoramaStitcher)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

add_executable(stitch src/stitch.cpp)
target_link_libraries(stitch ${OpenCV_LIBS})