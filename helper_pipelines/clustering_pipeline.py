# libraries
# import igraph
import random
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score
import itertools
# from igraph import RainbowPalette
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import time
from scipy import stats
import seaborn as sns
import pickle
from collections import OrderedDict
import sys
import os

# my own code

from helper_pipelines import choose_seeds

from helper_pipelines import read_coauthorship as read
from helper_pipelines.choose_seeds import *
from helper_pipelines.eigengap_calculator import *

# These are the parameters. Should make them readable from command line arguments.
K = 3
ALPHA = 0.4
STAR = False
N = 21
FILEPATH = 'data/dblp/coauthorship_c_style.txt'
REP = 100
PATHFORIMAGE = 'networkx_coauthorship_k3.png'
VECTOR_PATH = 'data/dblp/coauthorship_vectors.txt'

plt.rcParams.update({'font.size': 18})

def cluster_stats(x, estimator, clusters, node_to_cluster):
    print("Cluster stats:")
    print ("Size:", cluster_size(node_to_cluster, clusters))
    print("Radii:", cluster_radius(x, estimator, clusters))
    print("Distribution of distance from center across clusters:", distribution_of_dist_from_center())
    print("Betweeness centrality across clusters:", betweeness_centrality())

def cluster_size(node_to_cluster, clusters):
    size = {cluster:0 for cluster in clusters}
    for node in node_to_cluster:
        size[node_to_cluster[node]] += 1
    return size

def distribution_of_dist_from_center():
    return "NOT YET IMPLEMENTED"

def betweeness_centrality():
    return "NOT YET IMPLEMENTED"

def cluster_radius(x, estimator, y):
    # help from https://datascience.stackexchange.com/questions/32753/find-cluster-diameter-and-associated-cluster-points-with-kmeans-clustering-scik/32776
    y_kmeans = estimator.predict(x)
    #empty dictionaries

    clusters_centroids=dict()
    clusters_radii= dict()

    '''looping over clusters and calculate Euclidian distance of
    each point within that cluster from its centroid and
    pick the maximum which is the radius of that cluster'''

    for cluster in list(set(y)):

        clusters_centroids[cluster]=list(zip(estimator.cluster_centers_[:, 0],estimator.cluster_centers_[:,1]))[cluster]
        clusters_radii[cluster] = max([np.linalg.norm(np.subtract(i,clusters_centroids[cluster])) for i in zip(x[y_kmeans == cluster, 0],x[y_kmeans == cluster, 1])])

    # print (clusters_radii)
    #Visualising the clusters and cluster circles

    # fig, ax = plt.subplots(1,figsize=(7,5))
    #
    # plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
    # art = mpatches.Circle(clusters_centroids[0],clusters_radii[0], edgecolor='r',fill=False)
    # ax.add_patch(art)
    #
    # plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
    # art = mpatches.Circle(clusters_centroids[1],clusters_radii[1], edgecolor='b',fill=False)
    # ax.add_patch(art)
    #
    # plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')
    # art = mpatches.Circle(clusters_centroids[2],clusters_radii[2], edgecolor='g',fill=False)
    # ax.add_patch(art)
    #
    # #Plotting the centroids of the clusters
    # plt.scatter(estimator.cluster_centers_[:, 0], estimator.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')
    #
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('data/dblp/kmeans.jpg',dpi=300)

    return clusters_radii


def cluster(vectors, k, graph, cluster):
    X = np.array(list(vectors.values()))
    # X = read.create_hardcoded_stars(0.5,20)

    labels = KMeans(n_clusters=k, random_state=1).fit_predict(X)
    # print(kmeans.labels_) # This seems to be correct
    # print (kmeans.cluster_centers_)

    # print (graph.number_of_nodes())
    for node in graph.nodes:
        graph.nodes[node]["cluster"] = labels[node]

    node_to_cluster = {}
    # nodes = list(graph.keys())
    # nodes.sort()
    # print(graph)
    for node in graph:
        node_to_cluster[node] = labels[node]

        # # vectors[node].reshape(1, -1)
        # # print(node, vectors[node])
        # vector = np.array(vectors[node]).reshape(1, -1)
        # cluster_label = kmeans.predict(vector)[0]
        # node_to_cluster[node] = cluster_label

    # cluster_stats(X, kmeans, kmeans.labels_, node_to_cluster)

    return node_to_cluster

def spectral_cluster_star(X, k, graph):
    sc = SpectralClustering(n_clusters = k, random_state=1, affinity='precomputed').fit(X)
    print(sc.labels_)

    for node in graph.nodes:
        graph.node[node]["cluster"] = sc.labels_[node]

    node_to_cluster = {}
    # nodes = list(graph.keys())
    # nodes.sort()
    # print(graph)
    for node in graph:
        node_to_cluster[node] = sc.labels_[node-1]

    return node_to_cluster

def spectral_cluster(k, graph):
    '''
    runs spectral clustering on graph and saves the cluster index in the cluster attribute of each node
    '''
    # help from https://stackoverflow.com/questions/23684746/spectral-clustering-using-scikit-learn-on-graph-generated-through-networkx
    node_list = list(graph.nodes())
    adj_matrix = nx.to_numpy_matrix(graph, nodelist=node_list) #Converts graph to an adj matrix with adj_matrix[i][j] represents weight between node i,j.

    labels = SpectralClustering(affinity = 'precomputed', assign_labels="discretize",random_state=0,n_clusters=k).fit_predict(adj_matrix)
    print(labels)

    for node in node_list:
        graph.nodes[node]["cluster"] = labels[node]

    node_to_cluster = {}
    # nodes = list(graph.keys())
    # nodes.sort()
    # print(graph)
    for node in node_list:
        node_to_cluster[node] = labels[node]

    return node_to_cluster

# def color_nodes(k, g, clusters, filename):
#     # Color the nodes according to their cluster, then plot
#
#     pal = RainbowPalette(n=k)
#
#     color_dict = {}
#     for index, cluster_label in enumerate(clusters.values()):
#         if cluster_label not in color_dict:
#             color_dict[cluster_label] = pal[int(cluster_label)]
#
#     g.vs['label'] = list(range(g.vcount()))
#     out = igraph.plot(g,layout=g.layout('kk'), vertex_color = [color_dict[cluster] for cluster in g.vs["cluster"]])
#     out.save(filename)
#
# def make_graph(graph, cluster):
#     # all_ids = sorted(list(set(itertools.chain.from_iterable((e['from'],e['to']) for e in entries))))
#     # raw_id_to_id = {raw:v for v,raw in enumerate(all_ids)}
#
#     g = igraph.Graph(len(graph.keys()))
#     # print(graph.keys())
#     g.vs["my_id"] = list(graph.keys())
#     # print(g.vs.find(name="2"))
#
#     for node in graph:
#         # print(node)
#         this_v = g.vs.find(my_id=node)
#         this_v["cluster"] = cluster[node]
#         for neighbor in graph[node]:
#             that_v = g.vs.find(my_id=neighbor)
#             that_v["cluster"] = cluster[neighbor]
#             g.add_edge(this_v, that_v)
#
#     return g


def networkx_color(graph, k):
    pal = RainbowPalette(n=k)
    color_map = []
    for node in graph:
        print(graph.node[node]["cluster"])
        graph.node[node]["color"] = pal[int(graph.node[node]["cluster"])]
        print(graph.node[node]["color"])
        color_map.append(pal[int(graph.node[node]["cluster"])])

    return color_map


# code adapted from https://towardsdatascience.com/k-means-clustering-with-scikit-learn-6b47a369a83c
def elbow_method(vectors, min_k, max_k, step):
    '''
    Make elbow graph to choose k value
    '''
    X = np.array(list(vectors.values()))

    distortions = []
    for i in range(min_k, max_k):
        print("On k value " + str(i))
        kmeans = KMeans(n_clusters=i, random_state=1).fit(X)
        distortions.append(kmeans.inertia_)
        print (kmeans.inertia_)

    # plot
    print(distortions)
    plt.plot(range(min_k, max_k), distortions, marker='o')
    plt.xticks(range(min_k, max_k))
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title("Information Access Clustering Elbow Plot")
    plt.show()

# def spectral_elbow_method(graph, min_k, max_k, step):
#     '''
#     Make elbow graph to choose k value for spectral clustering
#     '''
#     node_list = list(graph.nodes())
#     adj_matrix = nx.to_numpy_matrix(graph, nodelist=node_list) #Converts graph to an adj matrix with adj_matrix[i][j] represents weight between node i,j.
#
#     distortions = []
#     for i in range(min_k, max_k):
#         print("On k value " + str(i))
#         labels = SpectralClustering(affinity = 'precomputed', assign_labels="discretize",random_state=0,n_clusters=k).fit(adj_matrix)
#         distortions.append(kmeans.inertia_)
#         print (kmeans.inertia_)
#
#     # plot
#     print(distortions)
#     plt.plot(range(min_k, max_k), distortions, marker='o')
#     plt.xticks(range(min_k, max_k))
#     plt.xlabel('Number of clusters')
#     plt.ylabel('Distortion')
#     plt.title("Information Access Clustering Elbow Plot")
#     plt.show()

def visualize(graph, clusters):
    color_map = networkx_color(graph, K)
    print(clusters)
    print (time.time() - start)
    # make visualization of graph colored by clustering
    g = make_graph(graph, clusters)
    color_nodes(K, g, clusters, PATHFORIMAGE)
    pos = nx.spring_layout(graph)
    nx.draw(graph, with_labels=True, node_color = color_map)
    plt.draw()
    plt.show()

def plot_citations(g):
    nodes = list(g.nodes(data=True))
    clusters_total = {cluster: 0 for cluster in range(K)}
    clusters_count = {cluster: 0 for cluster in range(K)}
    cluster_total_count = {cluster: 0 for cluster in range(K)}
    no_cites = 0
    for node in nodes:
        data = node[1]
        citations = data["citation_count"]
        cluster = data["cluster"]
        cluster_total_count[cluster] += 1
        if citations >= 0:
            clusters_total[cluster] += citations
            clusters_count[cluster] += 1
        else:
            no_cites += 1
    citation_averages = []
    for cluster in clusters_total:
        if clusters_count[cluster] == 0:
            print("cluster " + str(cluster) + " had no members")
            citation_averages.append(0)
        else:
            citation_averages.append(clusters_total[cluster]/clusters_count[cluster])

    print(f"There are {no_cites} people without citation records")
    print(clusters_count)
    print(cluster_total_count)
    print(citation_averages)
    plt.plot(citation_averages)
    plt.show()

def plot_attribute_distributions(g, attribute, cluster_method):
    '''
    Plots the distribution of some numerical node attribute for nodes in each cluster.
    '''
    print("\nStarting analysis of " + attribute)
    fig = plt.figure(figsize=(12,10))
    nodes = list(g.nodes(data=True))
    clusters_total = {cluster: [] for cluster in range(K)}
    no_cites = 0
    no_cites_dict = {cluster: 0.0 for cluster in range(K)}
    cluster_size = {cluster: 0.0 for cluster in range(K)}
    for node in nodes:
        data = node[1]
        # print(data)
        value = data[attribute]
        cluster = data["cluster"]
        cluster_size[cluster] +=1
        try: # negative value implies the value was not found
            clusters_total[cluster].append(float(value))
        except:
            no_cites += 1
            no_cites_dict[cluster] +=1
    print(cluster_size)
    citation_averages = []
    for cluster in clusters_total:
        input = [int(i) for i in clusters_total[cluster]]
        # clusters_total[cluster].sort()
        # plt.hist(clusters_total[cluster], bins = int(2450/5))
        print(f"total nodes in cluster {cluster}: {cluster_size[cluster]}")
        percent = no_cites_dict[cluster]/cluster_size[cluster]
        print(f"percent with {attribute} in cluster {cluster}: {1-percent}")
        sns.distplot(input, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = str(cluster))
    # with open("output_files/output_strings.txt", 'a') as file:
    #     file.write("\n0 to 2")
    #     file.write(str(stats.ks_2samp(clusters_total[0], clusters_total[2])))
    #     file.write("1 to 2")
    #     file.write(str(stats.ks_2samp(clusters_total[1], clusters_total[2])))
    #     file.write("0 to 1")
    #     file.write(str(stats.ks_2samp(clusters_total[0], clusters_total[1])))

    print(len(clusters_total[0]), clusters_total[0])
    print(len(clusters_total[1]), clusters_total[1])
    print(len(clusters_total[2]), clusters_total[2])

    print("\n0 to 2")
    print(stats.ks_2samp(clusters_total[0], clusters_total[2]))
    print("1 to 2")
    print(stats.ks_2samp(clusters_total[1], clusters_total[2]))
    print ("0 to 1")
    print(stats.ks_2samp(clusters_total[0], clusters_total[1]))

    # with open("output_files/output_strings.txt", 'a') as file:
    #     file.write("\nkruskal-wallis, 3-clusters:\n")
    #     file.write(stats.kruskal(clusters_total[0], clusters_total[1], clusters_total[2]))

    print("\nkruskal-wallis, 3-clusters:")
    print(stats.kruskal(clusters_total[0], clusters_total[1], clusters_total[2]))
    if (K == 12):
        print("kruskal-wallis, 12-clusters:")
        print(stats.kruskal(clusters_total[0], clusters_total[1], clusters_total[2], clusters_total[3], clusters_total[4], clusters_total[5], clusters_total[6], clusters_total[7], clusters_total[8], clusters_total[9], clusters_total[10], clusters_total[11]))

    print(attribute + str(len(nodes) - no_cites))
    plt.xlim(-50000, 100000)
    # plt.ylim(0, 5000)
    plt.xlabel(attribute)
    plt.ylabel("PDF")
    plt.title("Density at "+ attribute +" for different clusters")
    # plt.show()
    plt.savefig("../output_files/" + attribute + " vs. " + cluster_method + ".png", bbox_inches='tight')

def plot_attribute_bar(graph, attribute, cluster_method):
    fig = plt.figure()
    ax = plt.subplot()
    nodes = list(graph.nodes(data=True))
    cluster_to_attrib = {cluster: [] for cluster in range(K)}
    for node in nodes:
        data = node[1]
        # print(node)
        attrib = data[attribute]
        if attrib != "not found":
            cluster = data["cluster"]
            cluster_to_attrib[cluster].append(attrib)

    # width = 0.35
    # fig, ax = plt.subplots()
    for cluster in cluster_to_attrib:
        # need list of x axis and height of y axis
        cluster_size = len(cluster_to_attrib[cluster])
        attrib_lst = cluster_to_attrib[cluster]
        freqs = {i: attrib_lst.count(i)/cluster_size for i in set(attrib_lst)}
        print(freqs)
        ax.bar(list(freqs.keys()), list(freqs.values()), label=cluster, alpha = 0.2, linewidth=1)

    plt.title("Frequency of "+ attribute +" across information access clusters")
    plt.xlabel(attribute)
    plt.ylabel("portion of cluster")
    ax.legend()
    plt.savefig("../output_files/plots/" + attribute + " vs. " + cluster_method + ".png")
    plt.clf()


def connected_components(graph):
    size_comps = [len(c) for c in sorted(nx.connected_components(graph), key=len, reverse=False)]
    print(size_comps)

def add_centrality(graph):
    '''
    Adds attributes representing network structure importance metrics to the nodes of graph.
    '''
    deg_centrality = nx.degree_centrality(graph)
    # between_centrality = nx.betweenness_centrality(graph)
    pagerank = nx.pagerank_numpy(graph)
    for node in deg_centrality:
        graph.nodes[node]["degree_centrality"] = deg_centrality[node]
        # graph.nodes[node]["betweeness_centrality"] = between_centrality[node]
        graph.nodes[node]["pagerank"] = pagerank[node]


def read_in_clusters(filename):
    '''
    Reads in clusters from the cluster csv files.
    '''
    cluster_dict = OrderedDict()
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            line = line.split(",")
            # print(line)
            cluster_dict[line[0]] = int(line[1])
    return cluster_dict


def adj_rand_index(dict1, dict2):
    '''
    Calculates the adjusted rand index between two clusterings.
    '''
    # print(graph1.nodes())
    # spec_graph_clusters = nx.get_node_attributes(graph1,'cluster')
    # info_access_graph = nx.get_node_attributes(graph2,'cluster')
    if (list(dict1.keys()) == list(dict2.keys())):
        vals = list(dict1.values())
        # print(dict2.keys())
        print(adjusted_rand_score(list(dict1.values()), list(dict2.values()))) #check these actually align in the same order
    else:
        print("Order of two dicts is wrong")

def plot_all_attributes(graph, cluster_method):
    '''
    Plots distributions of various attributes over nodes in each cluster.
    '''
    plot_attribute_distributions(graph, "followers_count", cluster_method)
    # plot_attribute_bar(graph, "gender", cluster_method)
    # plot_attribute_distributions(graph, "phd_rank", cluster_method)
    # plot_attribute_distributions(graph, "degree_centrality", cluster_method)
    # plot_attribute_distributions(graph, "betweeness_centrality", cluster_method)
    # plot_attribute_distributions(graph, "pagerank", cluster_method)
    # plot_attribute_distributions(graph, "job_rank", cluster_method)

def dblp_citations_pipeline(cluster_method):
    if cluster_method=="info_access":
        graph = read.make_network_with_citations("data/dblp/coauthorship_dblp_ids.txt", "data/dblp/dblp_id_citations")
        add_centrality(graph)
        # print(nx.number_connected_components(graph))
        # print(graph.number_of_nodes())
        # connected_components(graph)
        vectors = read.read_in_vectors("data/dblp/old_coauthorship_vectors_48.txt") #need to make vectors
        clusters = cluster(vectors, K, graph)
        plot_citations(graph)
        plot_all_attributes(graph, "information access")

        # plot_attribute_bar(graph, "gender")
        # with open("data/dblp/info_access_graph_46_pickle", "ab") as f:
        #     pickle.dump(graph, f)
        # read.writeout_clusters(graph, "data/dblp/info_access_46_clusters.csv")
    elif cluster_method=="spectral":
        graph = read.make_network_with_citations("data/dblp/coauthorship_dblp_ids.txt", "data/dblp/dblp_id_citations")
        add_centrality(graph)
        clusters = spectral_cluster(K, graph)
        plot_citations(graph)
        # plot_all_attributes(graph, "spectral")
        # with open("data/dblp/spectral_graph_46_pickle", "ab") as f:
        #     pickle.dump(graph, f)
        # read.writeout_clusters(graph, "data/dblp/spectral_clusters_46.csv")
    else:
        spectral_dict = read_in_clusters("data/dblp/spectral_clusters_46.csv")
        info_access_dict = read_in_clusters("data/dblp/info_access_46_clusters.csv")
        adj_rand_index(spectral_dict, info_access_dict)

# def seed_pipeline(p):
#     graph = read.make_network_with_citations("data/dblp/coauthorship_dblp_ids.txt", "data/dblp/dblp_id_citations")
#     # add_centrality(graph)
#     seeds = random_seeds(graph, p)
#     read.write_graph(graph, seeds, "data/dblp/random_seed_edgelist.txt", True)
def plot_p(filename):
    with open(filename, "r") as f:
        aris = f.readlines()
        ari_vals = []
        for ari in aris:
            if ari != "\n":
                ari_vals.append(float(ari))
    fig = plt.figure(figsize=(12,10))
    x_vals = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    plt.plot(x_vals, ari_vals)
    plt.xlabel("p")
    plt.ylabel("Adjusted Random Index")
    plt.title("ARI vs. p")
    # plt.show()
    plt.savefig("data/dblp/plots/ari_p.png", bbox_inches='tight')

# def main():
#     start =  time.time()
#     if STAR:
#         graph = read.create_networkx_star(N, True)
#     else:
#         graph = read.make_networkx(FILEPATH)
#
#     # Read in vectors
#     vectors = read.read_in_vectors(VECTOR_PATH)
#
#     # cluster vectors using kmeans
#     clusters = cluster(vectors, K, graph)
#
#     # print (clusters)

def pipeline_before_vectors_cc():
    '''
    Runs the first half of the basic pipeline on only the largest connected component,
    until vectors need to be generated in C++
    '''
    graph = read.make_network_with_ids("data/dblp/coauthorship_dblp_ids.txt", "data/dblp/dblp_id_citations")
    largest_cc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest_cc).copy()
    add_centrality(graph)
    graph = nx.convert_node_labels_to_integers(graph) # replaces dblp id label with integer label. First label is 0.
    with open("data/dblp/cc_network_pickle", "wb") as f:
        pickle.dump(graph, f)
    read.write_graph(graph, [], "data/dblp/cc_edgelist.txt", False)

def pipeline_after_vectors_cc(vector_file):
    '''
    Runs the second half of the basic pipeline on only the largest connected component,
    after vectors have been generated in C++
    '''
    with open("data/dblp/cc_network_pickle", "rb") as f:
        graph = pickle.load(f)
    vectors = read.read_in_vectors(vector_file)
    clusters = cluster(vectors, K, graph)
    # read.writeout_clusters(graph, "data/dblp/cc_95_clusters.csv")
    plot_all_attributes(graph, "information access")
    clusters = spectral_cluster(K, graph)
    # read.writeout_clusters(graph, "data/dblp/spectral_3_clusters.csv")
    plot_all_attributes(graph, "spectral")

def pipeline_before_vectors():
    '''
    Runs the first half of the basic pipeline, until vectors need to be generated in C++
    '''
    graph = read.make_network_with_ids("data/dblp/coauthorship_dblp_ids.txt", "data/dblp/dblp_id_citations")
    add_centrality(graph)
    graph = nx.convert_node_labels_to_integers(graph) # replaces dblp id label with integer label. First label is 0.
    with open("data/dblp/network_pickle", "wb") as f:
        pickle.dump(graph, f)
    read.write_graph(graph, [], "data/dblp/edgelist.txt", False)

def pipeline_after_vectors(vector_file):
    '''
    Runs the second half of the basic pipeline, after vectors have been generated in C++
    '''
    # os.chdir('..')
    # print(os.getcwd())
    # os.chdir('output_files')
    print(os.getcwd())
    with open("../output_files/pickled_graph_quantifiers_added", "rb") as f:
        graph = pickle.load(f)
    vectors = read.read_in_vectors(vector_file)
    print("\n================INFORMATION ACCESS==================")
    cluster_method = "iac"
    clusters = cluster(vectors, K, graph, cluster_method)
    plot_all_attributes(graph, "information access")
    print("\n================SPECTRAL==================")
    cluster_method = "spectral"
    clusters = spectral_cluster(K, graph)
    plot_all_attributes(graph, "spectral")

def cc_info_access_elbow_pipeline(vector_file):
    with open("output_files/pickled_graph", "rb") as f:
        graph = pickle.load(f)
    # choose_spectral_k(graph)
    vectors = read.read_in_vectors(vector_file)
    elbow_method(vectors, 1, 10, 1)

def seed_before_vector_pipeline(seed_strategy, p):
    '''
    Runs the first half of the basic pipeline on only the largest connected component,
    until vectors need to be generated in C++. Provides list of seeds to create vectors with.
    '''
    graph = read.make_network_with_ids("data/dblp/coauthorship_dblp_ids.txt", "data/dblp/dblp_id_citations")
    largest_cc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest_cc).copy()
    add_centrality(graph)
    graph = nx.convert_node_labels_to_integers(graph) # replaces dblp id label with integer label. First label is 0.
    with open("data/dblp/cc_network_pickle", "wb") as f:
        pickle.dump(graph, f)
    seeds = []
    if seed_strategy == "random":
        seeds = random_seeds(graph, p)
    elif seed_strategy == "between":
        seeds = centrality_seeds(graph, p, "betweeness_centrality")
    elif seed_strategy == "degree":
        seeds = centrality_seeds(graph, p, "degree_centrality")
    elif seed_strategy == "pagerank":
        seeds = centrality_seeds(graph, p, "pagerank")
    read.write_graph(graph, seeds, f"data/dblp/cc_{seed_strategy}_seed_edgelist.txt", True)

def seed_after_vectors_cc(vector_file):
    '''
    Runs the second half of the basic pipeline on only the largest connected component,
    after vectors with a particular subset of seeds have been generated in C++
    '''
    with open("data/dblp/cc_network_pickle", "rb") as f:
        graph = pickle.load(f)
    vectors = read.read_in_seed_vectors(vector_file)
    clusters = cluster(vectors, 3, graph)
    plot_all_attributes(graph, "information access")
    clusters = spectral_cluster(3, graph)
    plot_all_attributes(graph, "spectral")

def seed_compare_cc(vector_file, p):
    with open("data/dblp/cc_network_pickle", "rb") as f:
        graph = pickle.load(f)
    vectors = read.read_in_seed_vectors(vector_file)
    clusters = cluster(vectors, 3, graph)
    read.writeout_clusters(graph, f"data/dblp/seed_clusters_{p}.csv")
    seed_clusters = read_in_clusters(f"data/dblp/seed_clusters_{p}.csv")
    full_clusters = read_in_clusters("data/dblp/cc_4_clusters.csv")
    adj_rand_index(seed_clusters, full_clusters)

def compare_clusters():
    info_access_clusters = read_in_clusters("data/dblp/cc_95_clusters.csv")
    spectral_clusters = read_in_clusters("data/dblp/spectral_12_clusters.csv")
    adj_rand_index(info_access_clusters, spectral_clusters)

if __name__ == "__main__":
    '''
    Any "before" pipeline should be run before the C++ code. It will build a networkx graph
    based off of the dblp data, pickle the graph, and write out an edgelist of the graph correctly
    formatted to be used as input to the C++ code.

    Any "after" pipeline should be run once you have generated vectors using the C++ code.
    It will unpickle the networkx graph and then run some kind of clustering and analysis,
    depending on which after pipeline it is.

    Any "cc" pipeline is run only on the largest fully connected component of the dblp network
    '''
    with open("data/dblp/dblp_id_citations", "rb") as f:
        graph = pickle.load(f)
    for node in graph.nodes:
        print(graph.nodes[node])
        print(type(graph.nodes[node]["citation_count"]))
    print("\n\n\n\n\n\n\n\n")
    if (sys.argv[1] == "before_vectors"):
        # This is the stanard before pipeline for the full dblp network with full seeds
        pipeline_before_vectors()
    elif (sys.argv[1] == "before_vectors_cc"):
        # This is the same as the previous but only with the largest fully connected component
        pipeline_before_vectors_cc()
    elif (sys.argv[1] == "after_vectors"):
        # This is the standard after pipeline for the full dblp network with full seeds.
        # It does both info access and spectral clustering, and saves plots comparing each clustering
        # to each of the data features.
        pipeline_after_vectors("../output_files/cc_vectors.txt")
    elif (sys.argv[1] == "after_vectors_cc"):
        # This does the same as above but only on the largest fully connected component.
        pipeline_after_vectors_cc("data/dblp/cc_vectorspoint4.txt")
    elif (sys.argv[1] == "cc_info_elbow"):
        # This creates an elbow plot for info access clustering on the largest connected component only
        # (Change the vector file path to generate an elbow plot based off a different alpha value)
        cc_info_access_elbow_pipeline("data/dblp/cc_vectorspoint05.txt")
    elif (sys.argv[1] == "before_vectors_seed_cc"):
        # This runs the first half of the pipeline on the largest fully connected component.
        # It also chooses the top p (which you can input through sys.argv[2]) seeds according
        # to the metric given in the first argument, and includes them in the edgelist file
        # so that p-dimensional vectors can be generated with the C++ code.
        seed_before_vector_pipeline("degree", int(sys.argv[2]))
    elif (sys.argv[1] == "after_vectors_seed_cc"):
        # This is the after pipeline that goes with the previous before pipeline. It runs info
        # access clustering and spectral clustering on the network using the p-dimensional vectors
        # and creates analysis plots.
        seed_after_vectors_cc("data/dblp/cc_degree_seed_vectors.txt")
    elif (sys.argv[1] == "find_p_after_vectors"):
        # This pipeline is for creating a plot to compare the p-dimensional vector clustering
        # to the n-dimensional vector clustering across different p values.
        seed_compare_cc("data/dblp/cc_degree_seed_vectors.txt", int(sys.argv[2]))
        print("\n")
    elif (sys.argv[1] == "compare_clusters"):
        # This pipeline compares spectral clustering to info access clustering using the adjusted
        # rand index.
        compare_clusters()
    else:
        print ("Invalid option")





# plot_p("data/dblp/pagerank_p_ari.txt")


    # dblp_citations_pipeline("info_access") # Change filename
    # seed_pipeline(6)
    # spectral_dict = read_in_clusters("data/dblp/spectral_clusters_46.csv")
    # info_access_dict = read_in_clusters("data/dblp/info_access_46_clusters.csv")
    # adj_rand_index(spectral_dict, info_access_dict)

    # graph = read.make_networkx(FILEPATH)
    # print(graph.size(), graph.number_of_nodes())
    # graph = read.make_network_with_citations("data/dblp/coauthorship_dblp_ids.txt", "data/dblp/dblp_id_citations")
    # read.write_graph(graph, [], "data/dblp/c_style_48.txt", False)
    # vectors = read.read_in_vectors("data/dblp/vectors_48.txt")
    # clusters = cluster(vectors, 3, graph)
    # read.writeout_clusters(graph, "data/dblp/info_access_clusters_48.csv")
