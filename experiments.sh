#!/bin/bash
for i in {1..10}; do

    mpiexec -n 4 MPIRotateImage hubble_image.png output_images 45
    ./RotateImage hubble_image.png output_images/serial_hubble_image.png 45
done
