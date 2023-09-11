cmake_minimum_required(VERSION 3.12)

project(YourProjectName)

# Find OpenCV package
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find nlohmann_json package
find_package(nlohmann_json REQUIRED)

# Find Google Test package
find_package(GTest REQUIRED)

# Find OpenCL package
find_package(OpenCL REQUIRED)

# Add your source files
add_executable(YourExecutable testsuit.cpp inferEng.cpp)

# Link libraries
target_link_libraries(YourExecutable ${OpenCV_LIBS} nlohmann_json::nlohmann_json GTest::GTest GTest::Main OpenCL::OpenCL)
