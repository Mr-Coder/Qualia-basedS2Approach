cmake_minimum_required(VERSION 3.14)
project(MathReasoningCore VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_TESTS "Build test executables" ON)
option(BUILD_BENCHMARKS "Build benchmark executables" ON)
option(ENABLE_OPTIMIZATION "Enable compiler optimizations" ON)

# Set optimization flags
if(ENABLE_OPTIMIZATION)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
    elseif(MSVC)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
    endif()
endif()

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(CORE_SOURCES
    src/complexity_classifier.cpp
    src/ird_engine.cpp
    src/deep_implicit_engine.cpp
    src/mlr_processor.cpp
    src/pattern_matcher.cpp
    src/utils.cpp
)

# Create core library
add_library(math_reasoning_core STATIC ${CORE_SOURCES})
target_include_directories(math_reasoning_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Python bindings
pybind11_add_module(math_reasoning_cpp
    src/python_bindings.cpp
    ${CORE_SOURCES}
)

# Link libraries
target_link_libraries(math_reasoning_cpp PRIVATE pybind11::module)

# Set properties for Python module
set_target_properties(math_reasoning_cpp PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Installation
install(TARGETS math_reasoning_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
)

# Export configuration
install(TARGETS math_reasoning_cpp
    LIBRARY DESTINATION ${Python_SITELIB}/math_reasoning
)