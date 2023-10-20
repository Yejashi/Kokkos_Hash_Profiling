cmake_minimum_required (VERSION 3.12)

project(HashProfiling LANGUAGES CXX)

find_package(Kokkos REQUIRED)

set(BIN_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(barebones src/profiling_kokkos_barebones.cpp)
add_executable(murmur3 src/profiling_kokkos_murmur3.cpp)

set_target_properties(
    barebones
    murmur3
    PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY ${BIN_DIRECTORY}
)

target_include_directories(murmur3 PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(barebones Kokkos::kokkos)
target_link_libraries(murmur3 Kokkos::kokkos)

set(CMAKE_CXX_FLAGS "${CXXFLAGS}")
