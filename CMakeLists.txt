cmake_minimum_required(VERSION 2.8)
project(GraphSlamCpp)

set(CMAKE_BUILD_TYPE "Debug") # Debug Release
set(CMAKE_CXX_FLAGS "-std=c++11 -g -Wall")



set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -Wno-reorder" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -DNDEBUG -Wno-reorder -O2" CACHE STRING "" FORCE)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# third party libs
# eigen
find_package(Eigen REQUIRED)
include_directories(${EIGEN_INCLUDE_DIR})

find_package(Boost COMPONENTS filesystem  REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
if(OpenCV_FOUND)
    message(STATUS "        OpenCV Version: ${OpenCV_VERSION}")
    message(STATUS "        OpenCV Include: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "        OpenCV Library: ${OpenCV_LIBS}")
endif(OpenCV_FOUND)

# glog
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

# G2O
# find_package(CSparse REQUIRED)
# include_directories(${CSPARSE_INCLUDE_DIR})
find_package(Cholmod REQUIRED)
include_directories(${CHOLMOD_INCLUDE_DIR})
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIR})
if(G2O_FOUND)
    message(STATUS "        G2O Root:    ${G2O_ROOT}")
    message(STATUS "        G2O Include: ${G2O_INCLUDE_DIR}")
    message(STATUS "        G2O Library: ${G2O_LIBS}")
endif(G2O_FOUND)

# sophus

# FIND_PACKAGE(OpenMP REQUIRED)
# if(OPENMP_FOUND)
#     message("OPENMP FOUND")
#     ADD_DEFINITIONS(-DUSE_OPENMP)
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif()

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
add_subdirectory(src)

