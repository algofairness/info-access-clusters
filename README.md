when writing paths to directories, include the slash at the end
  i.e, to write the path to directory Bar, use .../Foo/Bar/






# information-access-clustering

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
