cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(linemod)
# set(the_description "RGBD algorithms")

add_definitions(-DCVAPI_EXPORTS)
aux_source_directory(./src/  rgbd_src)
add_library(rgbd  SHARED ${rgbd_src})
target_link_libraries(rgbd ${OpenCV_LIBS})

add_subdirectory(./src/App/)

