cmake_minimum_required(VERSION 3.15)

project(play_on_marker)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(OpenCV REQUIRED )
find_package(nlohmann_json 3.2.0 REQUIRED)




include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${SFML_INCLUDE_DIRS})



add_executable(play_on_marker main.cpp)
target_link_libraries( play_on_marker ${OpenCV_LIBS} nlohmann_json::nlohmann_json ${SFML_LIBRARIES})
