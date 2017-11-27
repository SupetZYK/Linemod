cmake_minimum_required(VERSION 2.8.0)
project(object_recognition_renderer)

find_package(Boost REQUIRED)
# find_package(OpenCV 3 REQUIRED)

option(USE_GLUT "Use GLUT instead of OSMesa" OFF)

if (${USE_GLUT})
find_package(GLUT REQUIRED)
set(${PROJECT_NAME}_depends GLUT)
endif()

include_directories(SYSTEM ${Boost_INCLUDE_DIRS}
                           ${OpenCV_INCLUDE_DIRS}
)
message("OpenCV FOUND IN" ${OpenCV_INCLUDE_DIRS})
add_subdirectory(src)
