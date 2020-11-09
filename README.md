# information-access-clustering

This repository consists of code that runs the full Information Access Clustering pipeline, from running simulations of the independent cascade model of information propagation to information access and spectral clustering and plotting the results.

**General experimentation pipeline:**

1. Pipelines "build_*" to build a relevant graph pickle and edgelist for simulations: in main_pipelines.
2. Simulations to generate vector files for each of the alpha values: with run.sh.
3. Gap, Silhouette, and/or Elbow methods to find K: in main_pipelines.
4. Pipeline "after_vectors" to run clustering information access and spectral clustering methods and generate plots: in main_pipelines.

**Execution Files:**
1. run.sh: bash script for running "build_*" scripts, simulations, and after_vectors pipeline.
2. run_k.sh: for finding the K hyperparameter.

Please edit the bash scripts with the specific methods you'd like to run, as well as the relevant hyperparameters 
the methods use in main_pipelines (specified inside).

**On the naming of graphs:**

"twitch_PDF_K2_i1_10000_views_vs_information_access.png" means:
- "twitch": unique identifier string for "twitch" dataset.
- "PDF": Probability Density Function graph.
- "K2": used two clusters.
- "i1": "i" stands for individual (instead of "g" as in group), which means the graph was run using an individual alpha
value as opposed to a loop in the bash script -- please feel free to disregard. It is followed by the alpha value
after the decimal point (eg. 0.05 -> "05").
- "10000": number of simulations used to create the vector file.
- "views": specific attribute used to graph the PDF.
- "information_access": clustering method used (alternative: "spectral").

**Notes:**

- We chose the biggest connected component of a graph at hand to run the simulations. In doing so, if the graph is
directed and/or has parallel edges, we treated it as undirected and/or having single edges at the stage of choosing
the biggest connected component, despite the edge directions. However, when running the simulations we treated it 
as having directed and/or parallel edges. This way, no vertex is isolated, making the clusters meaningful, and the 
graph supports the Independent Cascade Model in a way that's authentic to the real-life conditions of information spread.
- Similarly, when performing spectral clustering, since the method runs on a symmetric adjacency matrix, we converted the
graphs to ones with undirected, single edges.
- Each "build_*" script creates a pickled graph and edgelist for the largest connected component of the original graph.

This code was written for the paper "Clustering via Information Access in a Network," which is available here: https://arxiv.org/abs/2010.12611. 
