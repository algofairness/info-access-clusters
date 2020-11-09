# information-access-clustering

Execution files:

run.sh:
  This file will run the full pipeline to generate analysis plots for the full dblp network (including islands). To run:

  ./run.sh ALPHA REPS

  I recommend using reps >= 10,000 to get good results. I've also been using alpha around 0.4, though the clusters are fairly consistent across different alpha values.

cc_run.sh
  This file does the exact same thing as run.sh, except on only the largest fully connected component of the dblp network.

choose_p.sh
  This file runs clustering on the fully connected component of the dblp network with only the top pagerank seeds. It generates a file of adjusted rand indices between the p-dimensional vector clusterings and the full vector clusterings.

seed_subset_run.sh:
  This file runs the full clustering pipeline for the largest fully connected component of the dblp network with a subset of the nodes as seeds for the vectors. It is currently set up to run clustering with the seeds that have the top degree centralities.


Other:
There are several other pipelines that you can run that don't have execution files. I would recommend starting with clustering_pipeline.py and looking at the bottom of the file. There are many different options that can be entered from the command line to run different pipelines. There are comments in the file describing what each does.
