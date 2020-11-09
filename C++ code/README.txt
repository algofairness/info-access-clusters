To use:

Make sure your data is in the format show in seed_dataset_example.txt, and described here:

The first line should be the number of nodes in your network, followed by either
0 or 1 to indicate whether the network is directed. (0 for undirected, 1 for directed)
The following lines should be an edgelist, with each line representing an edge between two nodes.
If the network is directed, it should be in the format:
from  to
If you are running the simulation with a subset of the nodes as seeds, then at the end of the file there should be a line starting with "s" that lists the seed nodes,
separated by tabs. All separators within lines in the whole file should be tabs.

Once your data is in the correct format, run run.sh. You will be prompted to put in
the path to your data file, the path where you want the output saved, the probability
of information spreading between any two connected nodes (if you don't know, 0.1 is a pretty standard
estimate) and the number of times you want the simulation repeated (if the network isn't too
big 10,000 is a good number, but obviously you can decide to lower it or raise it depending on a
speed/accuracy tradeoff).

If you are using a subset of the nodes as seeds, then the first row of the output will list the seeds. The following rows will be vectors.
Each row is the vector for that row number's node. Each column represents a seed,
so that the item at row i column j is the probability that node i receives information
from seed j (according to the seed ordering in row 0).
