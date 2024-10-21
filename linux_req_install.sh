#!/bin/bash

# Update package lists
apt-get update

# Install required system libraries for OpenGL
apt-get install -y libgl1-mesa-glx freeglut3-dev

# Install Python dependencies
pip install -r requirements.txt
