cmake_minimum_required(VERSION 3.16)
project(cosmo_viz LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(OpenMP)

# --- Common library ---
add_library(common STATIC
    src/common/Config.cpp
    src/common/HDF5IO.cpp
    src/common/Kernel.cpp
    src/common/KernelLUT.cpp
    src/common/SnapshotReader.cpp
    src/common/SmoothingLength.cpp
)
target_include_directories(common PUBLIC src)
target_link_libraries(common PUBLIC ${HDF5_C_LIBRARIES})
# HDF5_INCLUDE_DIRS may be empty on some module systems; find it from the library path
if(HDF5_INCLUDE_DIRS)
    target_include_directories(common PUBLIC ${HDF5_INCLUDE_DIRS})
else()
    # Derive include dir from library path
    get_filename_component(_hdf5_lib_dir "${HDF5_C_LIBRARY_hdf5}" DIRECTORY)
    get_filename_component(_hdf5_prefix "${_hdf5_lib_dir}" DIRECTORY)
    target_include_directories(common PUBLIC "${_hdf5_prefix}/include")
    message(STATUS "HDF5 include dir derived: ${_hdf5_prefix}/include")
endif()
# SmoothingLength.cpp uses OpenMP pragmas; propagate the runtime to consumers.
if(OpenMP_CXX_FOUND)
    target_link_libraries(common PUBLIC OpenMP::OpenMP_CXX)
endif()

# --- Gridder ---
add_executable(gridder
    src/gridder/main.cpp
    src/gridder/Grid.cpp
    src/gridder/Depositor.cpp
)
target_link_libraries(gridder PRIVATE common MPI::MPI_CXX)

# --- Tests ---
enable_testing()
add_executable(test_kernel_lut test/test_kernel_lut.cpp)
target_link_libraries(test_kernel_lut PRIVATE common)
add_test(NAME kernel_lut COMMAND test_kernel_lut)

add_executable(test_bvh
    test/test_bvh.cpp
    src/sph_renderer/BVH.cpp
)
target_link_libraries(test_bvh PRIVATE common)
add_test(NAME bvh COMMAND test_bvh)

# --- Renderer ---
add_executable(renderer
    src/renderer/main.cpp
    src/renderer/Camera.cpp
    src/renderer/GridReader.cpp
    src/renderer/RayTracer.cpp
    src/renderer/ImageWriter.cpp
)
target_link_libraries(renderer PRIVATE common MPI::MPI_CXX)
if(OpenMP_CXX_FOUND)
    target_link_libraries(renderer PRIVATE OpenMP::OpenMP_CXX)
endif()

# --- SPH Renderer (direct ray tracing, no grid intermediate) ---
add_executable(sph_renderer
    src/sph_renderer/main.cpp
    src/sph_renderer/ParticleStore.cpp
    src/sph_renderer/ParticleLoader.cpp
    src/sph_renderer/BVH.cpp
    src/sph_renderer/SphRayTracer.cpp
    src/sph_renderer/ImageWriter.cpp
    src/renderer/Camera.cpp
)
target_link_libraries(sph_renderer PRIVATE common MPI::MPI_CXX)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sph_renderer PRIVATE OpenMP::OpenMP_CXX)
endif()
