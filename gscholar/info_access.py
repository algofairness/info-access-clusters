import networkx as nx
import numpy as np
import sys

from sklearn.cluster import KMeans

# Parameters for finding K:
MIN_CLUSTERS = 2
MAX_CLUSTERS = 11

def main():
    vector_file_path = sys.argv[1]
    outfilename = sys.argv[2]
    clusternames = sys.argv[3]
    
    print("reading vectors...")
    file = open(vector_file_path, "r")
    nodes = file.readlines()
    vectors = {}
    for index, line in enumerate(nodes):
        line = line.split(",")
        node = index
        vectors[node] = []
        count = 0
        #print("the length of the line is ", len(line))
        for prob in line:
            vectors[node].append(float(prob))

    n = index + 1

    X = np.array(list(vectors.values()))
    
    outfile = open(outfilename, "w")
    for k in range(MIN_CLUSTERS, MAX_CLUSTERS, 1):
        outfile.write("for " + str(k) + " clusters:\n")
        
        print("info access # of clusters: " + str(k))
        labels = KMeans(n_clusters=k, random_state=1).fit_predict(X)
        
        file = open(clusternames[:-4] + "_" + str(k) + ".txt", "w")
        for id in range(0, n):
            file.write(str(id) + "\t" + str(labels[id]) + "\n")
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
