#!/usr/bin/env bash
echo "Hello, $(whoami)!"

# Runs the elbow method for K-Means Clustering based on the VECTOR_FILE (vectors produced
# by the C++ code) corresponding to a specific alpha value. One should run the simulations
# and change the filename in the constant to generate the plot for different
# alpha hyperparameters.

python3 k_methods_star.py
#python3 main_pipelines.py elbow_method
