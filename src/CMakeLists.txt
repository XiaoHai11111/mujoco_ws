cmake_minimum_required(VERSION 3.10)

project(dummy)

find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(dummy_idyn dummy_inverse_dynamics.cpp)

target_link_libraries(dummy_idyn 
    ${PINOCCHIO_LIBRARIES} 
    ${EIGEN3_LIBRARIES}
)
