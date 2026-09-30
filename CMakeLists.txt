cmake_minimum_required(VERSION 3.16)
project(stateless_matrix_whitener LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(smw_optimizer STATIC
    src/optimizer.cpp
)
target_include_directories(smw_optimizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(smw_optimizer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_options(smw_optimizer PRIVATE -Wall -Wextra)

add_executable(quadratic_demo tools/quadratic_demo.cpp)
target_link_libraries(quadratic_demo PRIVATE smw_optimizer)
target_compile_options(quadratic_demo PRIVATE -Wall -Wextra)

enable_testing()

add_executable(test_optimizer tests/test_optimizer.cpp)
target_link_libraries(test_optimizer PRIVATE smw_optimizer)
target_include_directories(test_optimizer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
)
target_compile_options(test_optimizer PRIVATE -Wall -Wextra)

add_test(NAME unit_tests COMMAND test_optimizer)
add_test(NAME quadratic_demo_converges COMMAND quadratic_demo)

# Negative tests: these invocations supply invalid configuration (illegal
# shape, non-positive stability parameter, negative learning rate, NaN
# gradient) and therefore must FAIL (non-zero exit) without updating the
# parameters. The "zero_case" is the zero regularization case and must fail.
add_test(NAME reject_rows_gt_columns COMMAND quadratic_demo --m 3 --n 2)
set_tests_properties(reject_rows_gt_columns PROPERTIES WILL_FAIL TRUE)

add_test(NAME reject_dimension_above_32 COMMAND quadratic_demo --m 33 --n 33)
set_tests_properties(reject_dimension_above_32 PROPERTIES WILL_FAIL TRUE)

add_test(NAME reject_zero_regularization_zero_case
         COMMAND quadratic_demo 0.0 0.1)
set_tests_properties(reject_zero_regularization_zero_case PROPERTIES
                     WILL_FAIL TRUE)

add_test(NAME reject_negative_regularization
         COMMAND quadratic_demo -1e-8 0.1)
set_tests_properties(reject_negative_regularization PROPERTIES WILL_FAIL TRUE)

add_test(NAME reject_negative_learning_rate
         COMMAND quadratic_demo 1e-8 -0.5)
set_tests_properties(reject_negative_learning_rate PROPERTIES WILL_FAIL TRUE)

add_test(NAME reject_nan_gradient COMMAND quadratic_demo --nan)
set_tests_properties(reject_nan_gradient PROPERTIES WILL_FAIL TRUE)
