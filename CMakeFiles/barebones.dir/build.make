# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/bbogale/package_managers/spack/opt/spack/linux-rhel7-power9le/gcc-9.2.0/cmake-3.26.3-oquohsiekwsylf7gxuaplrbxzheyerns/bin/cmake

# The command to remove a file.
RM = /home/bbogale/package_managers/spack/opt/spack/linux-rhel7-power9le/gcc-9.2.0/cmake-3.26.3-oquohsiekwsylf7gxuaplrbxzheyerns/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/bbogale/hash_profiling/Kokkos_Hash_Profiling

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/bbogale/hash_profiling/Kokkos_Hash_Profiling

# Include any dependencies generated for this target.
include CMakeFiles/barebones.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/barebones.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/barebones.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/barebones.dir/flags.make

CMakeFiles/barebones.dir/src/profiling_kokkos_barebones.cpp.o: CMakeFiles/barebones.dir/flags.make
CMakeFiles/barebones.dir/src/profiling_kokkos_barebones.cpp.o: src/profiling_kokkos_barebones.cpp
CMakeFiles/barebones.dir/src/profiling_kokkos_barebones.cpp.o: CMakeFiles/barebones.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bbogale/hash_profiling/Kokkos_Hash_Profiling/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/barebones.dir/src/profiling_kokkos_barebones.cpp.o"
	/home/bbogale/package_managers/spack/var/spack/environments/kokkos_gpu/.spack-env/view/bin/kokkos_launch_compiler /home/bbogale/package_managers/spack/var/spack/environments/kokkos_gpu/.spack-env/view/bin/nvcc_wrapper /apps/spack/10-10-2020/opt/spack/linux-rhel7-power9le/gcc-9.2.0/gcc-9.2.0-l2piuopjr3pzlfz6ybwp2c5wnhe37n3f/bin/g++ /apps/spack/10-10-2020/opt/spack/linux-rhel7-power9le/gcc-9.2.0/gcc-9.2.0-l2piuopjr3pzlfz6ybwp2c5wnhe37n3f/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/barebones.dir/src/profiling_kokkos_barebones.cpp.o -MF CMakeFiles/barebones.dir/src/profiling_kokkos_barebones.cpp.o.d -o CMakeFiles/barebones.dir/src/profiling_kokkos_barebones.cpp.o -c /home/bbogale/hash_profiling/Kokkos_Hash_Profiling/src/profiling_kokkos_barebones.cpp

CMakeFiles/barebones.dir/src/profiling_kokkos_barebones.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/barebones.dir/src/profiling_kokkos_barebones.cpp.i"
	/apps/spack/10-10-2020/opt/spack/linux-rhel7-power9le/gcc-9.2.0/gcc-9.2.0-l2piuopjr3pzlfz6ybwp2c5wnhe37n3f/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bbogale/hash_profiling/Kokkos_Hash_Profiling/src/profiling_kokkos_barebones.cpp > CMakeFiles/barebones.dir/src/profiling_kokkos_barebones.cpp.i

CMakeFiles/barebones.dir/src/profiling_kokkos_barebones.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/barebones.dir/src/profiling_kokkos_barebones.cpp.s"
	/apps/spack/10-10-2020/opt/spack/linux-rhel7-power9le/gcc-9.2.0/gcc-9.2.0-l2piuopjr3pzlfz6ybwp2c5wnhe37n3f/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bbogale/hash_profiling/Kokkos_Hash_Profiling/src/profiling_kokkos_barebones.cpp -o CMakeFiles/barebones.dir/src/profiling_kokkos_barebones.cpp.s

# Object files for target barebones
barebones_OBJECTS = \
"CMakeFiles/barebones.dir/src/profiling_kokkos_barebones.cpp.o"

# External object files for target barebones
barebones_EXTERNAL_OBJECTS =

bin/barebones: CMakeFiles/barebones.dir/src/profiling_kokkos_barebones.cpp.o
bin/barebones: CMakeFiles/barebones.dir/build.make
bin/barebones: /home/bbogale/package_managers/spack/var/spack/environments/kokkos_gpu/.spack-env/view/lib64/libkokkoscontainers.so.4.1.99
bin/barebones: /home/bbogale/package_managers/spack/var/spack/environments/kokkos_gpu/.spack-env/view/lib64/libkokkoscore.so.4.1.99
bin/barebones: /usr/lib64/libcuda.so
bin/barebones: /home/bbogale/package_managers/spack/opt/spack/linux-rhel7-power9le/gcc-9.2.0/cuda-12.1.1-y5gmzhqaoj7zceszc6mta3kjbeseky22/lib64/libcudart.so
bin/barebones: /apps/spack/10-10-2020/opt/spack/linux-rhel7-power9le/gcc-9.2.0/gcc-9.2.0-l2piuopjr3pzlfz6ybwp2c5wnhe37n3f/lib64/libgomp.so
bin/barebones: /lib64/libpthread.so
bin/barebones: /home/bbogale/package_managers/spack/var/spack/environments/kokkos_gpu/.spack-env/view/lib64/libkokkossimd.so.4.1.99
bin/barebones: CMakeFiles/barebones.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/bbogale/hash_profiling/Kokkos_Hash_Profiling/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bin/barebones"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/barebones.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/barebones.dir/build: bin/barebones
.PHONY : CMakeFiles/barebones.dir/build

CMakeFiles/barebones.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/barebones.dir/cmake_clean.cmake
.PHONY : CMakeFiles/barebones.dir/clean

CMakeFiles/barebones.dir/depend:
	cd /home/bbogale/hash_profiling/Kokkos_Hash_Profiling && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bbogale/hash_profiling/Kokkos_Hash_Profiling /home/bbogale/hash_profiling/Kokkos_Hash_Profiling /home/bbogale/hash_profiling/Kokkos_Hash_Profiling /home/bbogale/hash_profiling/Kokkos_Hash_Profiling /home/bbogale/hash_profiling/Kokkos_Hash_Profiling/CMakeFiles/barebones.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/barebones.dir/depend

