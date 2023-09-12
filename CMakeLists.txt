cmake_minimum_required(VERSION 3.10)

project(Vstream)

# Find required packages
find_package(GTest REQUIRED)
find_package(OpenCL REQUIRED)
find_package(OpenCV 4 REQUIRED)

# Include directories for header files
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/includes)

# Add your source files
add_executable(Vstream
    src/GlobalTestSuit.cpp
    src/infera.cpp
)

# Link libraries to your executable
target_link_libraries(Vstream
    ${GTEST_LIBRARIES}
    OpenCL::OpenCL
    ${OpenCV_LIBS}
)