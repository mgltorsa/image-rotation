# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mgltorsa/workspace/arch-parallel-computer/course-project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mgltorsa/workspace/arch-parallel-computer/course-project

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/mgltorsa/workspace/arch-parallel-computer/course-project/CMakeFiles /home/mgltorsa/workspace/arch-parallel-computer/course-project//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/mgltorsa/workspace/arch-parallel-computer/course-project/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named ScaleImage

# Build rule for target.
ScaleImage: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 ScaleImage
.PHONY : ScaleImage

# fast build rule for target.
ScaleImage/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/ScaleImage.dir/build.make CMakeFiles/ScaleImage.dir/build
.PHONY : ScaleImage/fast

#=============================================================================
# Target rules for targets named RotateImage

# Build rule for target.
RotateImage: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 RotateImage
.PHONY : RotateImage

# fast build rule for target.
RotateImage/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/RotateImage.dir/build.make CMakeFiles/RotateImage.dir/build
.PHONY : RotateImage/fast

#=============================================================================
# Target rules for targets named MPIRotateImage

# Build rule for target.
MPIRotateImage: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 MPIRotateImage
.PHONY : MPIRotateImage

# fast build rule for target.
MPIRotateImage/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/MPIRotateImage.dir/build.make CMakeFiles/MPIRotateImage.dir/build
.PHONY : MPIRotateImage/fast

#=============================================================================
# Target rules for targets named MPIOptimizedRotateImage

# Build rule for target.
MPIOptimizedRotateImage: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 MPIOptimizedRotateImage
.PHONY : MPIOptimizedRotateImage

# fast build rule for target.
MPIOptimizedRotateImage/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/MPIOptimizedRotateImage.dir/build.make CMakeFiles/MPIOptimizedRotateImage.dir/build
.PHONY : MPIOptimizedRotateImage/fast

#=============================================================================
# Target rules for targets named HelloWorld

# Build rule for target.
HelloWorld: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 HelloWorld
.PHONY : HelloWorld

# fast build rule for target.
HelloWorld/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/HelloWorld.dir/build.make CMakeFiles/HelloWorld.dir/build
.PHONY : HelloWorld/fast

HelloWorld.o: HelloWorld.cpp.o
.PHONY : HelloWorld.o

# target to build an object file
HelloWorld.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/HelloWorld.dir/build.make CMakeFiles/HelloWorld.dir/HelloWorld.cpp.o
.PHONY : HelloWorld.cpp.o

HelloWorld.i: HelloWorld.cpp.i
.PHONY : HelloWorld.i

# target to preprocess a source file
HelloWorld.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/HelloWorld.dir/build.make CMakeFiles/HelloWorld.dir/HelloWorld.cpp.i
.PHONY : HelloWorld.cpp.i

HelloWorld.s: HelloWorld.cpp.s
.PHONY : HelloWorld.s

# target to generate assembly for a file
HelloWorld.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/HelloWorld.dir/build.make CMakeFiles/HelloWorld.dir/HelloWorld.cpp.s
.PHONY : HelloWorld.cpp.s

mpi_rotation_image.o: mpi_rotation_image.cpp.o
.PHONY : mpi_rotation_image.o

# target to build an object file
mpi_rotation_image.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/MPIRotateImage.dir/build.make CMakeFiles/MPIRotateImage.dir/mpi_rotation_image.cpp.o
.PHONY : mpi_rotation_image.cpp.o

mpi_rotation_image.i: mpi_rotation_image.cpp.i
.PHONY : mpi_rotation_image.i

# target to preprocess a source file
mpi_rotation_image.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/MPIRotateImage.dir/build.make CMakeFiles/MPIRotateImage.dir/mpi_rotation_image.cpp.i
.PHONY : mpi_rotation_image.cpp.i

mpi_rotation_image.s: mpi_rotation_image.cpp.s
.PHONY : mpi_rotation_image.s

# target to generate assembly for a file
mpi_rotation_image.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/MPIRotateImage.dir/build.make CMakeFiles/MPIRotateImage.dir/mpi_rotation_image.cpp.s
.PHONY : mpi_rotation_image.cpp.s

mpi_rotation_image_optimized.o: mpi_rotation_image_optimized.cpp.o
.PHONY : mpi_rotation_image_optimized.o

# target to build an object file
mpi_rotation_image_optimized.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/MPIOptimizedRotateImage.dir/build.make CMakeFiles/MPIOptimizedRotateImage.dir/mpi_rotation_image_optimized.cpp.o
.PHONY : mpi_rotation_image_optimized.cpp.o

mpi_rotation_image_optimized.i: mpi_rotation_image_optimized.cpp.i
.PHONY : mpi_rotation_image_optimized.i

# target to preprocess a source file
mpi_rotation_image_optimized.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/MPIOptimizedRotateImage.dir/build.make CMakeFiles/MPIOptimizedRotateImage.dir/mpi_rotation_image_optimized.cpp.i
.PHONY : mpi_rotation_image_optimized.cpp.i

mpi_rotation_image_optimized.s: mpi_rotation_image_optimized.cpp.s
.PHONY : mpi_rotation_image_optimized.s

# target to generate assembly for a file
mpi_rotation_image_optimized.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/MPIOptimizedRotateImage.dir/build.make CMakeFiles/MPIOptimizedRotateImage.dir/mpi_rotation_image_optimized.cpp.s
.PHONY : mpi_rotation_image_optimized.cpp.s

rotation_image.o: rotation_image.cpp.o
.PHONY : rotation_image.o

# target to build an object file
rotation_image.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/RotateImage.dir/build.make CMakeFiles/RotateImage.dir/rotation_image.cpp.o
.PHONY : rotation_image.cpp.o

rotation_image.i: rotation_image.cpp.i
.PHONY : rotation_image.i

# target to preprocess a source file
rotation_image.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/RotateImage.dir/build.make CMakeFiles/RotateImage.dir/rotation_image.cpp.i
.PHONY : rotation_image.cpp.i

rotation_image.s: rotation_image.cpp.s
.PHONY : rotation_image.s

# target to generate assembly for a file
rotation_image.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/RotateImage.dir/build.make CMakeFiles/RotateImage.dir/rotation_image.cpp.s
.PHONY : rotation_image.cpp.s

scaling_image.o: scaling_image.cpp.o
.PHONY : scaling_image.o

# target to build an object file
scaling_image.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/ScaleImage.dir/build.make CMakeFiles/ScaleImage.dir/scaling_image.cpp.o
.PHONY : scaling_image.cpp.o

scaling_image.i: scaling_image.cpp.i
.PHONY : scaling_image.i

# target to preprocess a source file
scaling_image.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/ScaleImage.dir/build.make CMakeFiles/ScaleImage.dir/scaling_image.cpp.i
.PHONY : scaling_image.cpp.i

scaling_image.s: scaling_image.cpp.s
.PHONY : scaling_image.s

# target to generate assembly for a file
scaling_image.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/ScaleImage.dir/build.make CMakeFiles/ScaleImage.dir/scaling_image.cpp.s
.PHONY : scaling_image.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... HelloWorld"
	@echo "... MPIOptimizedRotateImage"
	@echo "... MPIRotateImage"
	@echo "... RotateImage"
	@echo "... ScaleImage"
	@echo "... HelloWorld.o"
	@echo "... HelloWorld.i"
	@echo "... HelloWorld.s"
	@echo "... mpi_rotation_image.o"
	@echo "... mpi_rotation_image.i"
	@echo "... mpi_rotation_image.s"
	@echo "... mpi_rotation_image_optimized.o"
	@echo "... mpi_rotation_image_optimized.i"
	@echo "... mpi_rotation_image_optimized.s"
	@echo "... rotation_image.o"
	@echo "... rotation_image.i"
	@echo "... rotation_image.s"
	@echo "... scaling_image.o"
	@echo "... scaling_image.i"
	@echo "... scaling_image.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

