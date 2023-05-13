#!/bin/bash

vpkg_require opencv
vpkg_require openmpi
vpkg_require gcc
vpkg_require cmake

cmake .
make