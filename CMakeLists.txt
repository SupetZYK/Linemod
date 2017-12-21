cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(OpenCV_LINEMOD)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(OpenCV_DIR "/usr/local/share/OpenCV")
find_package(OpenCV 3.2 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} include)
message("zyk found opencv: "${OpenCV_INCLUDE_DIRS})

message("OpenCV Libraries:"${OpenCV_LIBS})

add_subdirectory(linemod)
add_subdirectory(renderer)
