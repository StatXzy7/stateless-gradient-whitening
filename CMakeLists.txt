cmake_minimum_required(VERSION 3.16)
project(stateless_gradient_whitening CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

# Core library: zero third-party dependencies (only the C++ standard library).
add_library(stateless_whitener src/stateless_whitener.cpp)
target_include_directories(stateless_whitener PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Deterministic multi-step quadratic example.
add_executable(quadratic_demo examples/quadratic_demo.cpp)
target_link_libraries(quadratic_demo PRIVATE stateless_whitener)

enable_testing()
add_executable(test_stateless_whitener tests/test_stateless_whitener.cpp)
target_link_libraries(test_stateless_whitener PRIVATE stateless_whitener)
add_test(NAME stateless_whitener COMMAND test_stateless_whitener)
