#!/bin/bash
for i in {1..2}; do

    mpiexec -n 4 MPIRotateImage james_webb.png output_images 45
    ./RotateImage james_webb.png output_images/serial_james_webb.png 45
done
