# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.15.1/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.15.1/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/aglin/CLionProjects/dl85_folder/dl85/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/aglin/CLionProjects/dl85_folder/dl85/src/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/dl85.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/dl85.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/dl85.dir/flags.make

CMakeFiles/dl85.dir/main.cpp.o: CMakeFiles/dl85.dir/flags.make
CMakeFiles/dl85.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/aglin/CLionProjects/dl85_folder/dl85/src/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/dl85.dir/main.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dl85.dir/main.cpp.o -c /Users/aglin/CLionProjects/dl85_folder/dl85/src/main.cpp

CMakeFiles/dl85.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dl85.dir/main.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/aglin/CLionProjects/dl85_folder/dl85/src/main.cpp > CMakeFiles/dl85.dir/main.cpp.i

CMakeFiles/dl85.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dl85.dir/main.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/aglin/CLionProjects/dl85_folder/dl85/src/main.cpp -o CMakeFiles/dl85.dir/main.cpp.s

CMakeFiles/dl85.dir/codes/data.cpp.o: CMakeFiles/dl85.dir/flags.make
CMakeFiles/dl85.dir/codes/data.cpp.o: ../codes/data.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/aglin/CLionProjects/dl85_folder/dl85/src/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/dl85.dir/codes/data.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dl85.dir/codes/data.cpp.o -c /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/data.cpp

CMakeFiles/dl85.dir/codes/data.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dl85.dir/codes/data.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/data.cpp > CMakeFiles/dl85.dir/codes/data.cpp.i

CMakeFiles/dl85.dir/codes/data.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dl85.dir/codes/data.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/data.cpp -o CMakeFiles/dl85.dir/codes/data.cpp.s

CMakeFiles/dl85.dir/codes/dataContinuous.cpp.o: CMakeFiles/dl85.dir/flags.make
CMakeFiles/dl85.dir/codes/dataContinuous.cpp.o: ../codes/dataContinuous.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/aglin/CLionProjects/dl85_folder/dl85/src/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/dl85.dir/codes/dataContinuous.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dl85.dir/codes/dataContinuous.cpp.o -c /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/dataContinuous.cpp

CMakeFiles/dl85.dir/codes/dataContinuous.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dl85.dir/codes/dataContinuous.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/dataContinuous.cpp > CMakeFiles/dl85.dir/codes/dataContinuous.cpp.i

CMakeFiles/dl85.dir/codes/dataContinuous.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dl85.dir/codes/dataContinuous.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/dataContinuous.cpp -o CMakeFiles/dl85.dir/codes/dataContinuous.cpp.s

CMakeFiles/dl85.dir/codes/dataBinary.cpp.o: CMakeFiles/dl85.dir/flags.make
CMakeFiles/dl85.dir/codes/dataBinary.cpp.o: ../codes/dataBinary.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/aglin/CLionProjects/dl85_folder/dl85/src/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/dl85.dir/codes/dataBinary.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dl85.dir/codes/dataBinary.cpp.o -c /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/dataBinary.cpp

CMakeFiles/dl85.dir/codes/dataBinary.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dl85.dir/codes/dataBinary.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/dataBinary.cpp > CMakeFiles/dl85.dir/codes/dataBinary.cpp.i

CMakeFiles/dl85.dir/codes/dataBinary.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dl85.dir/codes/dataBinary.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/dataBinary.cpp -o CMakeFiles/dl85.dir/codes/dataBinary.cpp.s

CMakeFiles/dl85.dir/codes/dl85.cpp.o: CMakeFiles/dl85.dir/flags.make
CMakeFiles/dl85.dir/codes/dl85.cpp.o: ../codes/dl85.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/aglin/CLionProjects/dl85_folder/dl85/src/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/dl85.dir/codes/dl85.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dl85.dir/codes/dl85.cpp.o -c /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/dl85.cpp

CMakeFiles/dl85.dir/codes/dl85.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dl85.dir/codes/dl85.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/dl85.cpp > CMakeFiles/dl85.dir/codes/dl85.cpp.i

CMakeFiles/dl85.dir/codes/dl85.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dl85.dir/codes/dl85.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/dl85.cpp -o CMakeFiles/dl85.dir/codes/dl85.cpp.s

CMakeFiles/dl85.dir/codes/experror.cpp.o: CMakeFiles/dl85.dir/flags.make
CMakeFiles/dl85.dir/codes/experror.cpp.o: ../codes/experror.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/aglin/CLionProjects/dl85_folder/dl85/src/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/dl85.dir/codes/experror.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dl85.dir/codes/experror.cpp.o -c /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/experror.cpp

CMakeFiles/dl85.dir/codes/experror.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dl85.dir/codes/experror.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/experror.cpp > CMakeFiles/dl85.dir/codes/experror.cpp.i

CMakeFiles/dl85.dir/codes/experror.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dl85.dir/codes/experror.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/experror.cpp -o CMakeFiles/dl85.dir/codes/experror.cpp.s

CMakeFiles/dl85.dir/codes/globals.cpp.o: CMakeFiles/dl85.dir/flags.make
CMakeFiles/dl85.dir/codes/globals.cpp.o: ../codes/globals.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/aglin/CLionProjects/dl85_folder/dl85/src/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/dl85.dir/codes/globals.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dl85.dir/codes/globals.cpp.o -c /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/globals.cpp

CMakeFiles/dl85.dir/codes/globals.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dl85.dir/codes/globals.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/globals.cpp > CMakeFiles/dl85.dir/codes/globals.cpp.i

CMakeFiles/dl85.dir/codes/globals.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dl85.dir/codes/globals.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/globals.cpp -o CMakeFiles/dl85.dir/codes/globals.cpp.s

CMakeFiles/dl85.dir/codes/lcm_pruned.cpp.o: CMakeFiles/dl85.dir/flags.make
CMakeFiles/dl85.dir/codes/lcm_pruned.cpp.o: ../codes/lcm_pruned.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/aglin/CLionProjects/dl85_folder/dl85/src/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/dl85.dir/codes/lcm_pruned.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dl85.dir/codes/lcm_pruned.cpp.o -c /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/lcm_pruned.cpp

CMakeFiles/dl85.dir/codes/lcm_pruned.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dl85.dir/codes/lcm_pruned.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/lcm_pruned.cpp > CMakeFiles/dl85.dir/codes/lcm_pruned.cpp.i

CMakeFiles/dl85.dir/codes/lcm_pruned.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dl85.dir/codes/lcm_pruned.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/lcm_pruned.cpp -o CMakeFiles/dl85.dir/codes/lcm_pruned.cpp.s

CMakeFiles/dl85.dir/codes/query.cpp.o: CMakeFiles/dl85.dir/flags.make
CMakeFiles/dl85.dir/codes/query.cpp.o: ../codes/query.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/aglin/CLionProjects/dl85_folder/dl85/src/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/dl85.dir/codes/query.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dl85.dir/codes/query.cpp.o -c /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/query.cpp

CMakeFiles/dl85.dir/codes/query.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dl85.dir/codes/query.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/query.cpp > CMakeFiles/dl85.dir/codes/query.cpp.i

CMakeFiles/dl85.dir/codes/query.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dl85.dir/codes/query.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/query.cpp -o CMakeFiles/dl85.dir/codes/query.cpp.s

CMakeFiles/dl85.dir/codes/query_best.cpp.o: CMakeFiles/dl85.dir/flags.make
CMakeFiles/dl85.dir/codes/query_best.cpp.o: ../codes/query_best.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/aglin/CLionProjects/dl85_folder/dl85/src/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/dl85.dir/codes/query_best.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dl85.dir/codes/query_best.cpp.o -c /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/query_best.cpp

CMakeFiles/dl85.dir/codes/query_best.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dl85.dir/codes/query_best.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/query_best.cpp > CMakeFiles/dl85.dir/codes/query_best.cpp.i

CMakeFiles/dl85.dir/codes/query_best.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dl85.dir/codes/query_best.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/query_best.cpp -o CMakeFiles/dl85.dir/codes/query_best.cpp.s

CMakeFiles/dl85.dir/codes/query_totalfreq.cpp.o: CMakeFiles/dl85.dir/flags.make
CMakeFiles/dl85.dir/codes/query_totalfreq.cpp.o: ../codes/query_totalfreq.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/aglin/CLionProjects/dl85_folder/dl85/src/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/dl85.dir/codes/query_totalfreq.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dl85.dir/codes/query_totalfreq.cpp.o -c /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/query_totalfreq.cpp

CMakeFiles/dl85.dir/codes/query_totalfreq.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dl85.dir/codes/query_totalfreq.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/query_totalfreq.cpp > CMakeFiles/dl85.dir/codes/query_totalfreq.cpp.i

CMakeFiles/dl85.dir/codes/query_totalfreq.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dl85.dir/codes/query_totalfreq.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/query_totalfreq.cpp -o CMakeFiles/dl85.dir/codes/query_totalfreq.cpp.s

CMakeFiles/dl85.dir/codes/depthTwoComputer.cpp.o: CMakeFiles/dl85.dir/flags.make
CMakeFiles/dl85.dir/codes/depthTwoComputer.cpp.o: ../codes/depthTwoComputer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/aglin/CLionProjects/dl85_folder/dl85/src/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object CMakeFiles/dl85.dir/codes/depthTwoComputer.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dl85.dir/codes/depthTwoComputer.cpp.o -c /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/depthTwoComputer.cpp

CMakeFiles/dl85.dir/codes/depthTwoComputer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dl85.dir/codes/depthTwoComputer.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/depthTwoComputer.cpp > CMakeFiles/dl85.dir/codes/depthTwoComputer.cpp.i

CMakeFiles/dl85.dir/codes/depthTwoComputer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dl85.dir/codes/depthTwoComputer.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/depthTwoComputer.cpp -o CMakeFiles/dl85.dir/codes/depthTwoComputer.cpp.s

CMakeFiles/dl85.dir/codes/trie.cpp.o: CMakeFiles/dl85.dir/flags.make
CMakeFiles/dl85.dir/codes/trie.cpp.o: ../codes/trie.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/aglin/CLionProjects/dl85_folder/dl85/src/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object CMakeFiles/dl85.dir/codes/trie.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dl85.dir/codes/trie.cpp.o -c /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/trie.cpp

CMakeFiles/dl85.dir/codes/trie.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dl85.dir/codes/trie.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/trie.cpp > CMakeFiles/dl85.dir/codes/trie.cpp.i

CMakeFiles/dl85.dir/codes/trie.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dl85.dir/codes/trie.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/trie.cpp -o CMakeFiles/dl85.dir/codes/trie.cpp.s

CMakeFiles/dl85.dir/codes/dataBinaryPython.cpp.o: CMakeFiles/dl85.dir/flags.make
CMakeFiles/dl85.dir/codes/dataBinaryPython.cpp.o: ../codes/dataBinaryPython.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/aglin/CLionProjects/dl85_folder/dl85/src/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building CXX object CMakeFiles/dl85.dir/codes/dataBinaryPython.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dl85.dir/codes/dataBinaryPython.cpp.o -c /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/dataBinaryPython.cpp

CMakeFiles/dl85.dir/codes/dataBinaryPython.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dl85.dir/codes/dataBinaryPython.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/dataBinaryPython.cpp > CMakeFiles/dl85.dir/codes/dataBinaryPython.cpp.i

CMakeFiles/dl85.dir/codes/dataBinaryPython.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dl85.dir/codes/dataBinaryPython.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/dataBinaryPython.cpp -o CMakeFiles/dl85.dir/codes/dataBinaryPython.cpp.s

CMakeFiles/dl85.dir/codes/lcm_iterative.cpp.o: CMakeFiles/dl85.dir/flags.make
CMakeFiles/dl85.dir/codes/lcm_iterative.cpp.o: ../codes/lcm_iterative.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/aglin/CLionProjects/dl85_folder/dl85/src/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Building CXX object CMakeFiles/dl85.dir/codes/lcm_iterative.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dl85.dir/codes/lcm_iterative.cpp.o -c /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/lcm_iterative.cpp

CMakeFiles/dl85.dir/codes/lcm_iterative.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dl85.dir/codes/lcm_iterative.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/lcm_iterative.cpp > CMakeFiles/dl85.dir/codes/lcm_iterative.cpp.i

CMakeFiles/dl85.dir/codes/lcm_iterative.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dl85.dir/codes/lcm_iterative.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/lcm_iterative.cpp -o CMakeFiles/dl85.dir/codes/lcm_iterative.cpp.s

CMakeFiles/dl85.dir/codes/dataManager.cpp.o: CMakeFiles/dl85.dir/flags.make
CMakeFiles/dl85.dir/codes/dataManager.cpp.o: ../codes/dataManager.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/aglin/CLionProjects/dl85_folder/dl85/src/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_16) "Building CXX object CMakeFiles/dl85.dir/codes/dataManager.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dl85.dir/codes/dataManager.cpp.o -c /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/dataManager.cpp

CMakeFiles/dl85.dir/codes/dataManager.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dl85.dir/codes/dataManager.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/dataManager.cpp > CMakeFiles/dl85.dir/codes/dataManager.cpp.i

CMakeFiles/dl85.dir/codes/dataManager.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dl85.dir/codes/dataManager.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/dataManager.cpp -o CMakeFiles/dl85.dir/codes/dataManager.cpp.s

CMakeFiles/dl85.dir/codes/rCover.cpp.o: CMakeFiles/dl85.dir/flags.make
CMakeFiles/dl85.dir/codes/rCover.cpp.o: ../codes/rCover.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/aglin/CLionProjects/dl85_folder/dl85/src/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_17) "Building CXX object CMakeFiles/dl85.dir/codes/rCover.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dl85.dir/codes/rCover.cpp.o -c /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/rCover.cpp

CMakeFiles/dl85.dir/codes/rCover.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dl85.dir/codes/rCover.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/rCover.cpp > CMakeFiles/dl85.dir/codes/rCover.cpp.i

CMakeFiles/dl85.dir/codes/rCover.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dl85.dir/codes/rCover.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/rCover.cpp -o CMakeFiles/dl85.dir/codes/rCover.cpp.s

CMakeFiles/dl85.dir/codes/rCoverTotalFreq.cpp.o: CMakeFiles/dl85.dir/flags.make
CMakeFiles/dl85.dir/codes/rCoverTotalFreq.cpp.o: ../codes/rCoverTotalFreq.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/aglin/CLionProjects/dl85_folder/dl85/src/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_18) "Building CXX object CMakeFiles/dl85.dir/codes/rCoverTotalFreq.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dl85.dir/codes/rCoverTotalFreq.cpp.o -c /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/rCoverTotalFreq.cpp

CMakeFiles/dl85.dir/codes/rCoverTotalFreq.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dl85.dir/codes/rCoverTotalFreq.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/rCoverTotalFreq.cpp > CMakeFiles/dl85.dir/codes/rCoverTotalFreq.cpp.i

CMakeFiles/dl85.dir/codes/rCoverTotalFreq.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dl85.dir/codes/rCoverTotalFreq.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/rCoverTotalFreq.cpp -o CMakeFiles/dl85.dir/codes/rCoverTotalFreq.cpp.s

CMakeFiles/dl85.dir/codes/rCoverWeighted.cpp.o: CMakeFiles/dl85.dir/flags.make
CMakeFiles/dl85.dir/codes/rCoverWeighted.cpp.o: ../codes/rCoverWeighted.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/aglin/CLionProjects/dl85_folder/dl85/src/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_19) "Building CXX object CMakeFiles/dl85.dir/codes/rCoverWeighted.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dl85.dir/codes/rCoverWeighted.cpp.o -c /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/rCoverWeighted.cpp

CMakeFiles/dl85.dir/codes/rCoverWeighted.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dl85.dir/codes/rCoverWeighted.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/rCoverWeighted.cpp > CMakeFiles/dl85.dir/codes/rCoverWeighted.cpp.i

CMakeFiles/dl85.dir/codes/rCoverWeighted.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dl85.dir/codes/rCoverWeighted.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/aglin/CLionProjects/dl85_folder/dl85/src/codes/rCoverWeighted.cpp -o CMakeFiles/dl85.dir/codes/rCoverWeighted.cpp.s

# Object files for target dl85
dl85_OBJECTS = \
"CMakeFiles/dl85.dir/main.cpp.o" \
"CMakeFiles/dl85.dir/codes/data.cpp.o" \
"CMakeFiles/dl85.dir/codes/dataContinuous.cpp.o" \
"CMakeFiles/dl85.dir/codes/dataBinary.cpp.o" \
"CMakeFiles/dl85.dir/codes/dl85.cpp.o" \
"CMakeFiles/dl85.dir/codes/experror.cpp.o" \
"CMakeFiles/dl85.dir/codes/globals.cpp.o" \
"CMakeFiles/dl85.dir/codes/lcm_pruned.cpp.o" \
"CMakeFiles/dl85.dir/codes/query.cpp.o" \
"CMakeFiles/dl85.dir/codes/query_best.cpp.o" \
"CMakeFiles/dl85.dir/codes/query_totalfreq.cpp.o" \
"CMakeFiles/dl85.dir/codes/depthTwoComputer.cpp.o" \
"CMakeFiles/dl85.dir/codes/trie.cpp.o" \
"CMakeFiles/dl85.dir/codes/dataBinaryPython.cpp.o" \
"CMakeFiles/dl85.dir/codes/lcm_iterative.cpp.o" \
"CMakeFiles/dl85.dir/codes/dataManager.cpp.o" \
"CMakeFiles/dl85.dir/codes/rCover.cpp.o" \
"CMakeFiles/dl85.dir/codes/rCoverTotalFreq.cpp.o" \
"CMakeFiles/dl85.dir/codes/rCoverWeighted.cpp.o"

# External object files for target dl85
dl85_EXTERNAL_OBJECTS =

dl85: CMakeFiles/dl85.dir/main.cpp.o
dl85: CMakeFiles/dl85.dir/codes/data.cpp.o
dl85: CMakeFiles/dl85.dir/codes/dataContinuous.cpp.o
dl85: CMakeFiles/dl85.dir/codes/dataBinary.cpp.o
dl85: CMakeFiles/dl85.dir/codes/dl85.cpp.o
dl85: CMakeFiles/dl85.dir/codes/experror.cpp.o
dl85: CMakeFiles/dl85.dir/codes/globals.cpp.o
dl85: CMakeFiles/dl85.dir/codes/lcm_pruned.cpp.o
dl85: CMakeFiles/dl85.dir/codes/query.cpp.o
dl85: CMakeFiles/dl85.dir/codes/query_best.cpp.o
dl85: CMakeFiles/dl85.dir/codes/query_totalfreq.cpp.o
dl85: CMakeFiles/dl85.dir/codes/depthTwoComputer.cpp.o
dl85: CMakeFiles/dl85.dir/codes/trie.cpp.o
dl85: CMakeFiles/dl85.dir/codes/dataBinaryPython.cpp.o
dl85: CMakeFiles/dl85.dir/codes/lcm_iterative.cpp.o
dl85: CMakeFiles/dl85.dir/codes/dataManager.cpp.o
dl85: CMakeFiles/dl85.dir/codes/rCover.cpp.o
dl85: CMakeFiles/dl85.dir/codes/rCoverTotalFreq.cpp.o
dl85: CMakeFiles/dl85.dir/codes/rCoverWeighted.cpp.o
dl85: CMakeFiles/dl85.dir/build.make
dl85: CMakeFiles/dl85.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/aglin/CLionProjects/dl85_folder/dl85/src/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_20) "Linking CXX executable dl85"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dl85.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/dl85.dir/build: dl85

.PHONY : CMakeFiles/dl85.dir/build

CMakeFiles/dl85.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dl85.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dl85.dir/clean

CMakeFiles/dl85.dir/depend:
	cd /Users/aglin/CLionProjects/dl85_folder/dl85/src/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/aglin/CLionProjects/dl85_folder/dl85/src /Users/aglin/CLionProjects/dl85_folder/dl85/src /Users/aglin/CLionProjects/dl85_folder/dl85/src/cmake-build-debug /Users/aglin/CLionProjects/dl85_folder/dl85/src/cmake-build-debug /Users/aglin/CLionProjects/dl85_folder/dl85/src/cmake-build-debug/CMakeFiles/dl85.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dl85.dir/depend

