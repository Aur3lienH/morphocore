cmake_minimum_required(VERSION 3.18)
project(MorphoCore VERSION 1.0.0)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


option(ENABLE_CUDA "Enable CUDA support" ON)


if(ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    find_package(CUDAToolkit REQUIRED)

    set(CMAKE_CUDA_ARCHITECTURES "75;80;86")

    message(STATUS "CUDA enabled - Version: ${CUDAToolkit_VERSION}")

else()
    message(STATUS "CUDA disabled")
endif()


file(GLOB_RECURSE CPP_SOURCES "csrc/*.cpp")
file(GLOB_RECURSE CPP_HEADERS "csrc/*.hxx" "csrc/*.hpp" "csrc/*.h")

if(ENABLE_CUDA)
    file(GLOB_RECURSE CUDA_SOURCES "cuda/*.cu")
    file(GLOB_RECURSE CUDA_HEADERS "cuda/*.cuh" "cuda/*.hpp" "cuda/*.h")
endif()


add_executable(${PROJECT_NAME} ${ALL_SOURCES})


target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(${PROJECT_NAME} PRIVATE -O3)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})

message(STATUS "Python found - Version: ${Python3_VERSION}")
message(STATUS "Python executable: ${Python3_EXECUTABLE}")
message(STATUS "Python include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python libraries: ${Python3_LIBRARIES}")
message(STATUS "Python site packages: ${Python3_SITELIB}")

set(ALL_SOURCES ${CPP_SOURCES})

if(ENABLE_CUDA)
    list(APPEND ALL_SOURCES ${CUDA_SOURCES})
endif()

message(STATUS "Final sources: ${ALL_SOURCES}")

target_sources(${PROJECT_NAME} PRIVATE ${ALL_SOURCES})



message(STATUS "=== Build Configuration ===")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
if(ENABLE_CUDA)
    message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "===============================")


install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)