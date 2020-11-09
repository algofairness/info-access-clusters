import networkx as nx
import numpy as np
import sys
import math

from sklearn.cluster import SpectralClustering
#from sklearn.metrics import adjusted_rand_score
#from sklearn.metrics import pairwise_distances

# Range of K:
MIN_CLUSTERS = 2
MAX_CLUSTERS = 5

def main():
    filename = sys.argv[1]
    outfilename = sys.argv[2]
    clusternames = sys.argv[3]
    print("reading the graph...")
    
    file = open(filename, "r")
    lines = file.readlines()
    graph = nx.Graph()

    counter = 0
    for line in lines:
        if counter != 0:
            s = int(line.split(None, 2)[0])
            t = int(line.split(None, 2)[1])
            graph.add_node(s)
            graph.add_node(t)
            graph.add_edge(s, t)
            graph.add_edge(t, s) # for spectral use undirected
        counter +=1
        
    file.close()
    
    print(graph.number_of_nodes())
    n = graph.number_of_nodes()

    node_list = list(graph.nodes())
    # Converts graph to an adj matrix with adj_matrix[i][j] = weight between node i,j.
    #adj_matrix = nx.to_numpy_matrix(graph, nodelist=node_list, dtype=np.int32)
    adj_matrix = nx.to_scipy_sparse_matrix(graph, nodelist=node_list, dtype=np.int8)
    
    graph.clear()
    
    outfile = open(outfilename, "w")
    for k in range(MIN_CLUSTERS, MAX_CLUSTERS+1):
        outfile.write("for " + str(k) + " clusters:\n")
        
        print("spectral # of clusters: " + str(k))
        labels = SpectralClustering(affinity = 'precomputed', assign_labels="discretize",random_state=0,n_clusters=k).fit_predict(adj_matrix)
        #n_components = 10 * log(n) * k
        
        file = open(clusternames[:-4] + "_" + str(k) + ".txt", "w")
        for i in range(0, n):
            file.write(str(i) + "\t" + str(labels[i]) + "\n")
        file.close();
        
        buckets = [0] * k
        for i in range(0, n):
            buckets[labels[i]] += 1
        
        for i in range(0, k):
            outfile.write("%" + str(int(round(buckets[i] * 100 / n))) + " ")
        outfile.write("\n")
        
    outfile.close()
    return

if __name__ == "__main__":
    main()
