cmake_minimum_required(VERSION 3.15)
project(play_on_marker)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
#set(aruco_DIR /Users/luispedrooliveira/Downloads/aruco-3.1.11/build)

find_package( OpenCV REQUIRED )
find_package(nlohmann_json 3.2.0 REQUIRED)
#find_package(aruco REQUIRED )


include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${MPG123_INCLUDE_DIRS})
include_directories(${LIBAO_INCLUDE_DIRS})
#include_directories(/Users/luispedrooliveira/Downloads/aruco-3.1.11/src)



add_executable(play_on_marker main.cpp)
target_link_libraries( play_on_marker ${OpenCV_LIBS} nlohmann_json::nlohmann_json ${MPG123_LIBRARIES} ${LIBAO_LIBRARIES})
