#!/usr/bin/env bash
echo "Hello, $(whoami)!"

# Structure:
# 1. Runs the before_vectors pipeline for world_trade dataset to create a pickled networkx
#    object and an edgelist in the format compatible for running the simulations.
# 2. Compiles the C++ code.
# 3. Asks for experiment data for purposes of specified naming
#    and passing it further as arguments.
# 4. Runs the simulations and creates a vectors file for Information Access Clustering.
# 5. Runs the after_vectors pipeline based on the pickled graph and vectors file:
#    clustering methods and cluster analysis.

# Note:
#    The controllers of the after_vectors pipeline must be customized (for now)
#    in main_pipelines.py for each vector file (with a specific alpha value) and attribute.

# python3 main_pipelines.py before_vectors_world_trade

# g++ C++\ code/main.cpp -o C++\ code/main -std=c++11

echo 'Please enter the alpha value?'
read alpha
value=${alpha#*.}

echo 'Please enter the number of repetitions?'
read nr

echo "Run simulations for all seeds: y or n?"
read response

echo "output_files/dblp_edgelist.txt" "output_files/dblp_vectors_i${value}_${nr}.txt" $alpha $nr $response| ./C++\ code/main

# python3 main_pipelines.py after_vectors
