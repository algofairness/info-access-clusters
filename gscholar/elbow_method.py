import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd

from sklearn.cluster import KMeans
import matplotlib.cm as cm

# Parameters for finding K:
MIN_CLUSTERS = 1
MAX_CLUSTERS = 11
N_REFS = 4

# Elbow_method code by Hannah Beillinson, adapted from https://towardsdatascience.com/k-means-clustering-with-scikit-learn-6b47a369a83c.
def main():

    '''reads the vectors'''
    filename = sys.argv[1]
    file = open(filename, "r")
    nodes = file.readlines()
    vectors = {}
    for index, line in enumerate(nodes):
        line = line.split(",")
        # print("the length of the line is ", len(line))
        node = index
        vectors[node] = []
        for prob in line:
            vectors[node].append(float(prob))

    """Make elbow graph to choose k hyper-parameter for the clustering methods."""
    X = np.array(list(vectors.values()))

    distortions = []
    for i in range(MIN_CLUSTERS, MAX_CLUSTERS):
        print("On k value " + str(i))
        kmeans = KMeans(n_clusters=i, random_state=1).fit(X)
        distortions.append(kmeans.inertia_)
        print(kmeans.inertia_)

    # plot
    print(distortions)
    plt.plot(range(MIN_CLUSTERS, MAX_CLUSTERS), distortions, marker='o')
    plt.xticks(range(MIN_CLUSTERS, MAX_CLUSTERS))
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    #vector_file_name = vector_file[:-4].split("_")
    plt.title("Elbow Method")#"Information Access Clustering Elbow Plot (alpha = 0.{})").format(vector_file_name[-2][1:]))
    plt.tight_layout()
    plt.savefig("elbow_method")
    plt.close()
    return

if __name__ == "__main__":
    main()
