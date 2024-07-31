#!/bin/bash

#bash /lab/micah/Downloads/Anaconda3-2024.02-1-Linux-x86_64.sh

source .bashrc
conda activate yolov10

python3 /lab/micah/obj-det/test.py $1