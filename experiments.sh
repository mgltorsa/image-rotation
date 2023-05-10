#!/bin/bash

vpkg_require opencv
vpkg_require openmpi
vpkg_require gcc

export OPENCV_IO_MAX_IMAGE_PIXELS=1099511627776

for i in {1..10}; do

    ./RotateImage hubble_image.png output_images/serial_hubble_image.png 45
    mpiexec -n 4 MPIRotateImage hubble_image.png output_images 45
done
