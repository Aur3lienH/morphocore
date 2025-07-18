cmake_minimum_required(VERSION 3.18)
project(MorphoCore VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Python first
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Find pybind11
find_package(pybind11 REQUIRED)

# Auto-detect PyTorch from the current Python environment
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_FOUND
)

if(TORCH_FOUND EQUAL 0)
    message(STATUS "Found PyTorch cmake path: ${TORCH_PREFIX_PATH}")
    list(APPEND CMAKE_PREFIX_PATH ${TORCH_PREFIX_PATH})
    find_package(Torch REQUIRED)
    set(TORCH_AVAILABLE TRUE)
else()
    message(WARNING "PyTorch not found in Python environment")
    set(TORCH_AVAILABLE FALSE)
endif()

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

# Collect source files
file(GLOB_RECURSE CPP_SOURCES "cpp/*.cpp")
file(GLOB_RECURSE CPP_HEADERS "cpp/*.hxx" "cpp/*.hpp" "cpp/*.h")

set(ALL_SOURCES ${CPP_SOURCES})

if(ENABLE_CUDA)
    file(GLOB_RECURSE CUDA_SOURCES "cuda/*.cu")
    file(GLOB_RECURSE CUDA_HEADERS "cuda/*.cuh" "cuda/*.hpp" "cuda/*.h")
    list(APPEND ALL_SOURCES ${CUDA_SOURCES})
endif()

# Create Python module
add_library(morphocore_ext MODULE ${ALL_SOURCES})

# Set up include directories
target_include_directories(morphocore_ext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Python3_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(morphocore_ext PRIVATE 
    pybind11::module
    ${Python3_LIBRARIES}
)

# Add PyTorch if available
if(TORCH_AVAILABLE)
    target_include_directories(morphocore_ext PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(morphocore_ext PRIVATE ${TORCH_LIBRARIES})
    message(STATUS "PyTorch libraries: ${TORCH_LIBRARIES}")
endif()

if(ENABLE_CUDA)
    target_link_libraries(morphocore_ext PRIVATE 
        CUDA::cudart
        CUDA::cublas
    )
endif()

# Set module properties
set_target_properties(morphocore_ext PROPERTIES 
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# Compiler options
target_compile_options(morphocore_ext PRIVATE -O3)

# Install the Python module to the correct location
install(TARGETS morphocore_ext DESTINATION .)

message(STATUS "=== Build Configuration ===")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
if(ENABLE_CUDA)
    message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "Python: ${Python3_VERSION}")
if(TORCH_AVAILABLE)
    message(STATUS "PyTorch: ${Torch_VERSION}")
endif()
message(STATUS "===============================")