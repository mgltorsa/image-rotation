cmake_minimum_required(VERSION 2.8)
project( ImageProcessing )
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -O3")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( ScaleImage scaling_image.cpp )
target_link_libraries( ScaleImage ${OpenCV_LIBS} )
set_target_properties( ScaleImage PROPERTIES RUNTIME_OUTPUT_DIRECTORY "bin")


add_executable( RotateImage rotation_image.cpp)
target_link_libraries( RotateImage ${OpenCV_LIBS} )
set_target_properties( RotateImage PROPERTIES RUNTIME_OUTPUT_DIRECTORY "bin")


find_package( MPI REQUIRED )
add_executable( MPIRotateImage mpi_rotation_image.cpp )
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
target_link_libraries( MPIRotateImage ${OpenCV_LIBS} )
target_link_libraries( MPIRotateImage ${MPI_CXX_LIBRARIES})
set_target_properties( MPIRotateImage PROPERTIES RUNTIME_OUTPUT_DIRECTORY "bin")


add_executable( MPIOptimizedRotateImage mpi_rotation_image_optimized.cpp )
set_target_properties( MPIOptimizedRotateImage PROPERTIES RUNTIME_OUTPUT_DIRECTORY "bin")
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
target_link_libraries( MPIOptimizedRotateImage ${OpenCV_LIBS} )
target_link_libraries( MPIOptimizedRotateImage ${MPI_CXX_LIBRARIES})


add_executable( HelloWorld HelloWorld.cpp )
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
target_link_libraries( HelloWorld ${MPI_CXX_LIBRARIES})
target_link_libraries( HelloWorld ${OpenCV_LIBS} )
set_target_properties( HelloWorld PROPERTIES RUNTIME_OUTPUT_DIRECTORY "bin")

