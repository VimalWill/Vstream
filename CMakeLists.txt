cmake_minimum_required(VERSION 3.10)

project(Vstream)

# Find required packages
find_package(GTest REQUIRED)
find_package(OpenCL REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(GST REQUIRED gstreamer-1.0)
include_directories(${GST_INCLUDE_DIRS})

# Include directories for header files
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/includes)

# Add your source files
add_executable(TestSuit
    src/GlobalTestSuit.cpp
    src/infera.cpp
)

add_executable(Vstream 
    src/main.cpp 
    src/infera.cpp
)

# Link libraries to your executable
target_link_libraries(TestSuit
    ${GTEST_LIBRARIES}
    OpenCL::OpenCL
    ${OpenCV_LIBS}
)

target_link_libraries(Vstream 
    ${GST_LIBRARIES}
    ${GST_LDFLAGS}
    ${OpenCV_LIBS}
    OpenCL::OpenCL
    gstapp-1.0
)