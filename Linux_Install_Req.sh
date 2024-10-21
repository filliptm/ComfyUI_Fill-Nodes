#!/bin/bash

# Update package lists
sudo apt-get update

# Install required system libraries for OpenGL
sudo apt-get install -y libgl1-mesa-glx freeglut3-dev

# Install Python dependencies
pip install -r requirements.txt
