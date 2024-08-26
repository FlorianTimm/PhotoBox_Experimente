cmake_minimum_required(VERSION 3.0)
project(MyProject)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find libcamera
find_package(libcamera REQUIRED)
include_directories(${libcamera_INCLUDE_DIRS})

# Add your source files here
add_executable(MyExecutable main.cpp)

# Link against OpenCV and libcamera
target_link_libraries(MyExecutable ${OpenCV_LIBS} ${libcamera_LIBS})
