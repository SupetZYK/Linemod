AUX_SOURCE_DIRECTORY(. App_LIST)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
foreach( current_app ${App_LIST} )
    # I used a simple string replace, to cut off .cpp.
    get_filename_component(app_name  ${current_app}  NAME_WE)
    add_executable( ${app_name}  ${current_app})
    message("Now name is : " ${app_name})
    # Make sure YourLib is linked to each app
    target_link_libraries(${app_name}  ${PROJECT_NAME}_3d ${OpenCV_LIBRARIES} ${EXTRA_LIBRARIES} rgbd)
endforeach( current_app ${App_LIST} )

## add an executable to render views to a file
#ADD_EXECUTABLE(train_test train_test.cpp)

#SET_PROPERTY(TARGET train_test PROPERTY DEBUG_POSTFIX ${ASSIMP_DEBUG_POSTFIX})

#TARGET_LINK_LIBRARIES(train_test ${PROJECT_NAME}_3d ${OpenCV_LIBRARIES})

#SET_TARGET_PROPERTIES(train_test PROPERTIES
#                                     OUTPUT_NAME train_test
#)

#install(TARGETS train_test
#        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#)
