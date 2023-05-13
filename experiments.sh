#!/bin/bash

vpkg_require opencv
vpkg_require openmpi
vpkg_require gcc

export OPENCV_IO_MAX_IMAGE_PIXELS=$((2*1099511627776))

for i in {1..10}; do
    ./bin/RotateImage hubble_image.png output_images/serial_hubble_image.png 45
done

for i in {1..10}; do
    for j in {2..6}; do
        mpiexec -n $j bin/MPIRotateImage hubble_image.png output_images 45
    done
done
