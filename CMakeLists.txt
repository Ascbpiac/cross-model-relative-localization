cmake_minimum_required(VERSION 3.0)
project(MixVPR  LANGUAGES C CXX CUDA)

# set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g") 

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -Wall -g") 

add_subdirectory(hnswlib)

find_package(catkin REQUIRED COMPONENTS
    cv_bridge
    )

SET(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_DIR}/include)
link_directories(${OpenCV_DIR}/build/lib)
    
#tensorrt
set(tensorrt_dir "/usr/local/TensorRT-8.6.1.6")
include_directories(${tensorrt_dir}/include)
link_directories(${tensorrt_dir}/lib)
    
#cuda
set(CUDA_DIR  "/usr/local/cuda")
find_package(CUDA REQUIRED)
include_directories(${CUDA_DIR}/include)
link_directories(${CUDA_DIR}/lib64)

add_library(tensorrt_utils STATIC 
    /docker-ros/ws/D_VINS-main/D_VINS-main/loop_fusion/src/deep_net/tensorrt_utils.cpp
)

include_directories(
        /docker-ros/ws/D_VINS-main/D_VINS-main/loop_fusion/src/deep_net
        /usr/local/include/
        ./include
        ${catkin_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${tensorrt_dir}/include
        )

find_package(Ceres REQUIRED)

# 指定源文件
set(SOURCES
    mixvpr-hnsw.cpp
    # 如果你有其他源文件，可以添加到这里
)

# 创建可执行文件
add_executable(MixVPR ${SOURCES})

# 链接 hnswlib 库
# 如果你将 hnswlib 库编译为静态库或动态库，它应该通过以下方式链接
target_link_libraries(MixVPR PRIVATE hnswlib tensorrt_utils nvinfer cuda cudart ${catkin_LIBRARIES} ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})

# 如果你有其他依赖库，也需要在这里链接
# target_link_libraries(MixVPR PRIVATE other_lib)
