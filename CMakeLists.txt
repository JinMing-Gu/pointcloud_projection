cmake_minimum_required(VERSION 2.8.3)
project(pointcloud_projection)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -fext-numeric-literals")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -Wall -g")

# SET(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_CXX_FLAGS "-std=c++11")
# set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS}  -Wall  -march=native -g")
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -Wall   -march=native -g")

find_package(catkin REQUIRED COMPONENTS
    message_generation
    roscpp
    rospy
    std_msgs
    pcl_conversions
    pcl_ros
    cv_bridge
    image_transport
    sensor_msgs 
    geometry_msgs
    nav_msgs
    camera_info_manager
)

find_package(OpenCV 3.2.0 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

## Generate messages in the 'msg' folder
add_message_files(
    FILES
    # CustomPoint.msg
    # CustomMsg.msg
    GPS.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
    DEPENDENCIES
    std_msgs  # Or other packages containing msgs
    geometry_msgs
)



include_directories(
    include 
    ${catkin_INCLUDE_DIRS} 
    ${OpenCV_INCLUDE_DIRS} 
    ${PCL_INCLUDE_DIRS})

add_executable(livox_repub src/livox_repub.cpp)
target_link_libraries(livox_repub ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

add_executable(imu_process src/imu_processor/data_process_node.cpp src/imu_processor/data_process.cpp src/imu_processor/gyr_int.cpp )
target_link_libraries(imu_process ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

add_executable(pointcloud_projection_neu_avia_mynteye src/pointcloud_projection/pointcloud_projection_neu_avia_mynteye.cpp)
# add_dependencies(pointcloud_projection_neu_avia_mynteye ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(pointcloud_projection_neu_avia_mynteye ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

add_executable(pointcloud_projection_neu_horizon_zed2 src/pointcloud_projection/pointcloud_projection_neu_horizon_zed2.cpp)
# add_dependencies(pointcloud_projection_neu_horizon_zed2 ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(pointcloud_projection_neu_horizon_zed2 ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

add_executable(pointcloud_projection_uisee_hesai_cam src/pointcloud_projection/pointcloud_projection_uisee_hesai_cam.cpp)
# add_dependencies(pointcloud_projection_uisee_hesai_cam ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(pointcloud_projection_uisee_hesai_cam ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})