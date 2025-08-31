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
option(ENABLE_IPP "Enable Intel IPP support" ON)

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

# ✅ Simple Intel IPP Detection
set(IPP_FOUND FALSE)
if(ENABLE_IPP)
    # Try oneAPI first (most common on Arch Linux)
    if(DEFINED ENV{ONEAPI_ROOT})
        set(IPP_ROOT "$ENV{ONEAPI_ROOT}/ipp/latest")
    else()
        set(IPP_ROOT "/opt/intel/oneapi/ipp/latest")
    endif()
    
    # Check if IPP exists
    find_path(IPP_INCLUDE_DIR ipp.h PATHS "${IPP_ROOT}/include" "/usr/include")
    
    if(IPP_INCLUDE_DIR)
        # Set include directories - ippi.h is in ipp/ subdirectory
        set(IPP_INCLUDE_DIRS 
            ${IPP_INCLUDE_DIR}
            "${IPP_INCLUDE_DIR}/ipp"
        )
        
        # Verify ippi.h exists in the subdirectory
        if(EXISTS "${IPP_INCLUDE_DIR}/ipp/ippi.h")
            message(STATUS "Found ippi.h in: ${IPP_INCLUDE_DIR}/ipp/")
        else()
            message(WARNING "ippi.h not found in expected location: ${IPP_INCLUDE_DIR}/ipp/")
        endif()
        
        # Find libraries (assuming intel64)
        find_library(IPP_CORE NAMES ippcore PATHS "${IPP_ROOT}/lib/intel64" "/usr/lib")
        find_library(IPP_I NAMES ippi PATHS "${IPP_ROOT}/lib/intel64" "/usr/lib")
        find_library(IPP_S NAMES ipps PATHS "${IPP_ROOT}/lib/intel64" "/usr/lib")
        
        if(IPP_CORE AND IPP_I AND IPP_S)
            set(IPP_LIBRARIES ${IPP_CORE} ${IPP_I} ${IPP_S})
            set(IPP_FOUND TRUE)
            message(STATUS "Intel IPP found: ${IPP_INCLUDE_DIR}")
        endif()
    endif()
    
    if(NOT IPP_FOUND)
        message(WARNING "Intel IPP not found. Install with: yay -S intel-oneapi-basekit")
    endif()
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

# Add IPP include directories
if(IPP_FOUND)
    target_include_directories(morphocore_ext PRIVATE ${IPP_INCLUDE_DIRS})
    target_compile_definitions(morphocore_ext PRIVATE WITH_IPP=1)
endif()

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

# Add IPP libraries
if(IPP_FOUND)
    target_link_libraries(morphocore_ext PRIVATE ${IPP_LIBRARIES})
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
if(IPP_FOUND)
    message(STATUS "Intel IPP: ENABLED")
else()
    message(STATUS "Intel IPP: DISABLED")
endif()
message(STATUS "Python: ${Python3_VERSION}")
if(TORCH_AVAILABLE)
    message(STATUS "PyTorch: ${Torch_VERSION}")
endif()
message(STATUS "===============================")