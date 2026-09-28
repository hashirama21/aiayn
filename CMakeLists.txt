cmake_minimum_required(VERSION 3.16)
project(aiayn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(AIAYN_NATIVE "Compile with -march=native (fastest here, not portable)" ON)

add_library(aiayn INTERFACE)
target_include_directories(aiayn INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(aiayn INTERFACE -Wall -Wextra -Wpedantic -ffp-contract=off)
if(AIAYN_NATIVE)
  target_compile_options(aiayn INTERFACE -march=native)
endif()

add_executable(train_toy src/train_toy.cpp)
target_link_libraries(train_toy PRIVATE aiayn)

add_executable(aiayn_tests_float tests/test_all.cpp)
target_link_libraries(aiayn_tests_float PRIVATE aiayn)
add_executable(aiayn_tests_double tests/test_all.cpp)
target_link_libraries(aiayn_tests_double PRIVATE aiayn)
target_compile_definitions(aiayn_tests_double PRIVATE AIAYN_REAL=double)

enable_testing()
add_test(NAME tests_float COMMAND aiayn_tests_float)
add_test(NAME tests_double COMMAND aiayn_tests_double)
