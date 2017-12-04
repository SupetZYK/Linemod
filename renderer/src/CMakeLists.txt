FIND_PACKAGE(OpenGL)

find_package(PkgConfig REQUIRED)
find_package(ASSIMP REQUIRED)

# create the 2d rendering library
add_library(${PROJECT_NAME}_2d SHARED renderer2d.cpp)
target_link_libraries(${PROJECT_NAME}_2d ${OpenCV_LIBRARIES} rgbd)


# create the 3d rendering library
set(SOURCES model.cpp renderer3d.cpp utils.cpp)

find_package(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}
                           ${ASSIMP_INCLUDE_DIRS}
                           ${ASSIMP_INCLUDE_DIRS}/../
                           ${OpenCV_INCLUDE_DIR}
                           ${OPENGL_INCLUDE_DIR}
)
link_directories(${ASSIMP_LIBRARY_DIRS})

if (${USE_GLUT})
  add_definitions(-DUSE_RENDERER_GLUT=1)

  find_package(GLUT)
  include_directories(SYSTEM ${GLUT_INCLUDE_DIR})
  set(EXTRA_SOURCES renderer3d_impl_glut.cpp)
  set(EXTRA_LIBRARIES ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
else()
  set(EXTRA_SOURCES renderer3d_impl_osmesa.cpp)
  set(EXTRA_LIBRARIES OSMesa GLU)
endif()

add_library(${PROJECT_NAME}_3d  SHARED ${SOURCES} ${EXTRA_SOURCES})
target_link_libraries(${PROJECT_NAME}_3d ${EXTRA_LIBRARIES}
                                         ${OpenCV_LIBRARIES}
                                         ${ASSIMP_LIBRARIES}
                                         freeimage
)


## add an executable to render views to a file
#ADD_EXECUTABLE(view_generator main.cpp)

#SET_PROPERTY(TARGET view_generator PROPERTY DEBUG_POSTFIX ${ASSIMP_DEBUG_POSTFIX})

#TARGET_LINK_LIBRARIES(view_generator ${EXTRA_LIBRARIES})

#target_link_libraries(view_generator ${PROJECT_NAME}_3d)
#target_link_libraries(view_generator ${PROJECT_NAME}_2d)


# add some executables
add_subdirectory(apps)
add_subdirectory(zyk_apps)
