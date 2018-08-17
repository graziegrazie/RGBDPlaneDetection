cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

################### Variables. ####################
# Change if you want modify path or other values. #
###################################################

set(PROJECT_NAME rgbd_plane_detection)
# Output Variables
set(OUTPUT_DEBUG Debug/bin)
set(OUTPUT_RELEASE Release/bin)
# Folders files
set(CPP_DIR_1 include/MRF2.2)
set(CPP_DIR_2 ./)
set(CPP_DIR_3 ./)
set(HEADER_DIR_1 include/MRF2.2)
set(HEADER_DIR_2 )

############## CMake Project ################
#        The main options of project        #
#############################################

project(${PROJECT_NAME} CXX)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  roscpp
  rospy
  sensor_msgs
  std_msgs
  nodelet
  cv_bridge
  image_transport
  tf
  dynamic_reconfigure
  camera_info_manager
  plane_msgs
  geometry_msgs
  image_geometry
)

FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

include_directories(include ${catkin_INCLUDE_DIRS})
include_directories(include/MRF2.2)
include_directories(include/peac)
include_directories(/usr/include/eigen3)

catkin_package()

# Define Release by default.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
  message(STATUS "Build type not specified: Use Release by default.")
endif(NOT CMAKE_BUILD_TYPE)

# Definition of Macros
add_definitions(
   -D_DEBUG 
   -D_CONSOLE 
   -DUNICODE
   -D_UNICODE
)

############## Artefacts Output #################
# Defines outputs , depending Debug or Release. #
#################################################

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${OUTPUT_DEBUG}")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${OUTPUT_DEBUG}")
  set(CMAKE_EXECUTABLE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${OUTPUT_DEBUG}")
else()
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${OUTPUT_REL}")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${OUTPUT_REL}")
  set(CMAKE_EXECUTABLE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${OUTPUT_REL}")
endif()

################# Flags ################
# Defines Flags for Windows and Linux. #
########################################

if(MSVC)
   set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /W3 /MD /Od /Zi /EHsc")
   set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /W3 /GL /Od /Oi /Gy /Zi /EHsc")
endif(MSVC)
if(NOT MSVC)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fpermissive")
   if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
       set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
   endif()
endif(NOT MSVC)

################ Files ################
#   --   Add files to project.   --   #
#######################################

file(GLOB SRC_FILES
    ${CPP_DIR_1}/*.cpp
    ${CPP_DIR_2}/*.cpp
    ${CPP_DIR_3}/*.cpp
    ${HEADER_DIR_1}/*.h
    ${HEADER_DIR_2}/*.h
)

# Add executable to build.
add_executable(${PROJECT_NAME} 
   ${SRC_FILES}
)

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBS})
