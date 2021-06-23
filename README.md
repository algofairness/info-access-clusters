# Setting up the git repository 

Before cloning the git repository, download the `info_access_data_and_code` file structure from Google Drive.  Then, clone this repo into the `code` directory.  Note that the files in the git repo should be directly under the `code` directory.  So, this file should be accessible via `code/README.md` and *not* at `code/info-access-clusters/README.md`.   You may have to remove some intermediary directories to make this happen. 

# Running experiments

### Quick Start Guide

To run the experiment corresponding to `testing.ini` run the command `./easy-run.sh` from the `code` directory.  Note that this takes a little over 5 minutes on a laptop.  To run all experiments, use `./easy-run.sh -a`.  

The results from each experiment will be located in `code/output_files`. 

### Full description

The code in C++ code/, main.py, vector_analysis.py, and data_rep.py runs an
  independent cascade simulation to generate vectors, runs analysis on those vectors,
  and represents the results visually.

Run the code:
  - to run an experiment, type the following:
    python3 main.py config_files/testing.ini
    where testing.ini is the config file corresponding to your experiment

Directory Structure:
  - Make sure you have an output directory whose path matches the one in the config
    file variable [FILES][outputDir]. This is where your results will go
  - For organizational purposes you should have two directories above this repository
    named "data" and "results". These should hold any needed input data (such as the files
    referenced in config variables [FILES][inEdgesFile], [FILES][inNodesFile], [FILES][inHoldEdgesFile],
    [FILES][inHoldNodesFile], [FILES][inAnalysisDir])
  - When writing paths to directories in the config file, always include the slash
    at the end of the path to a directory (i.e, use .../Foo/Bar/ NOT .../Foo/Bar)

Config Files:
  - find config files in the config_files folder
  - see EXAMPLE.ini for a guide of how to use config files
  - generally try to have a unique [GENERAL][experimentName] for each file
  - NOTE: config files from previous experiments will not always work when run again.
    this is because as the pipeline grown, I add things to the config file. So always
    check the format of the most recent config file (EXAMPLE.ini) before running

When adding an analysis method, make sure to add:
  - variable to config file
  - global variable to main
  - clause to main.run_analysis()
  - analysis function in vector_analysis.py
  - clause to main.run_datarep()

# information-access-clustering info

This repository consists of code that runs the full Information Access Clustering pipeline:

1. Reconstructing graphs and edgelists for independent cascade simulations.
2. Performing simulations that generate vector files, given alpha values.
3. Tuning the hyperparameter K, the number of clusters for information access clustering, through Gap Statistic, Silhouette Analysis, and Elbow Method.
4. Running the Information Access Clustering and relevant statistical analyses.
5. Clustering the graph with existing methods for deeper analysis.

# Execution Files
1. run.sh: bash script for running "build_*" scripts, simulations, and after_vectors pipeline.
2. run_k.sh: for finding the K hyperparameter.

Please edit the bash scripts with the specific methods you'd like to run, as well as the relevant hyperparameters
the methods use in main_pipelines (specified inside).

# References to the Used Code Bases:

Tuning K:

- [Gap Statistic](https://anaconda.org/milesgranger/gap-statistic/notebook)
- [Silhouette Analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
- [Elbow Method](https://towardsdatascience.com/k-means-clustering-with-scikit-learn-6b47a369a83c)

Clustering:

- [Spectral Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html)
- [Fluid Communities](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.asyn_fluid.asyn_fluidc.html#networkx.algorithms.community.asyn_fluid.asyn_fluidc)
- [Role2Vec](https://github.com/benedekrozemberczki/karateclub)
- [Louvain](https://github.com/taynaud/python-louvain)
- [Core/Periphery](https://github.com/skojaku/core-periphery-detection/blob/7d924402caa935e0c2e66fca40457d81afa618a5/cpnet/Rombach.py)

Hypothesis Testing:

- [Kolmogorov-Smirnov](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)
- [Kruskal-Wallis](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html)
- [Fisher Exact](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html)

Additional Methods:

- [Connected Components](https://networkx.org/documentation/stable/reference/algorithms/component.html)
- [Adjusted Rand Index](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)


# information access regression info
When running regression experiments make sure to add the heatmap function in data_rep.py

TO DO:
  - make one heatmap function and just pass in analysis name
