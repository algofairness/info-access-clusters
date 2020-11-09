import matplotlib
import report_file_object_class as rfo
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import math
import os
import csv
import copy
import sys
import build_cosponsorship as cosponsorship
import build_dblp as dblp
import build_twitch_network as twitch
from sklearn.cluster import KMeans, SpectralClustering
from scipy import stats
from helper_pipelines import read_coauthorship as read
from helper_pipelines import clustering_pipeline as cp
from helper_pipelines import eigengap_calculator as eigen
import pyreadr
import pandas as pd

from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from cluster_consistency import cluster_labeling

# Gap parameters:
#   N_REFS
#   IDENTIFIER_STRING
#   MAX_CLUSTERS
#   VECTOR_FILE_INVARIANT
#   ALPHA_VALUES

# Silhouette parameters:
#   IDENTIFIER_STRING
#   MAX_CLUSTERS
#   VECTOR_FILE_INVARIANT
#   ALPHA_VALUES

# Elbow parameters:
#   IDENTIFIER_STRING
#   VECTOR_FILE_INVARIANT
#   MIN_CLUSTERS
#   MAX_CLUSTERS
#   ALPHA_VALUES

# Unique identifier string for each experiment.
# Must be either one word (eg. "twitch") or several words using a hyphen
# (eg. "strong-house"). Do NOT use "_" (underscore).
IDENTIFIER_STRING = "twitch"

# Parameters for finding K:
MIN_CLUSTERS = 1
MAX_CLUSTERS = 16
N_REFS = 4

# Pipeline after_vectors parameters:
K = 2  # Hyperparameter k.
ATTRIBUTE = "views"  # Node attribute, about which to create distribution or bar graphs.
INPUT_PICKLED_GRAPH = "output_files/twitch_pickle"  # Path to a pickled graph with nodes and relevant attributes.
VECTOR_FILE_INVARIANT = "output_files/twitch_vectors/{}_vectors_i{}_10000.txt".format(
    IDENTIFIER_STRING,
                                                                                              "{}")  # Invariant for the vector txt files for different alpha values.
REPORT_FILE_PATH = "output_files/{}_K{}_output_strings_{}.txt".format(IDENTIFIER_STRING, K,
                                                                      ATTRIBUTE)  # File to which write the results of the clustering experiment.
REPORT_FILE = rfo.ReportFileObject(REPORT_FILE_PATH)  # Instance of the report file (for convenience in writing to it).

# Alpha values for co-sponsorship data set:
# ALPHA_VALUES = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

# Alpha values for other data sets:
ALPHA_VALUES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                0.95]  # Must match the alpha values used for creating the vector files.

PLOT_PDF = 1  # Whether the attribute is for plotting a PDF or not; if 1, plot_attribute_distributions() will be run;
# if 0, plot_attribute_bar().
PDF_LOG = 1  # If PLOT_PDF = 1, when drawing a PDF, should we take log(attribute)?
LOG_BASE = 10  # If PDF_LOG = 1, what is the base for log used in creating a PDF.

# (Optional) parameters for consistency finding methods.
# Please set MAKE_CLUSTERING_CONSISTENT = 0 for now, until the consistency methods are finalized.
# MAKE_CLUSTERING_CONSISTENT = 0
# CONSISTENCY_TYPE = "qualitative"

# (Optional) parameter for dataset_pdf(), which plots the PDF of the entire dataset.
DATASET_LOG = 1  # If 0, input numbers for PDF are raw; if 1, log of LOG_BASE is taken of them.

# (Optional) Colors used for the graphs.
COLOR_PALETTE = ["#FFC107", "#1E88E5", "#2ECE54", "#EC09D7", "#DDEC4E", "#D81B50", "#CCD85D", "#3701FA", "#D39CA7", "#27EA9F", "#5D5613", "#DC6464"]
BAR_GRAPH_COLOR_PALETTE = ["#BA65A4", "#1A4D68"]

# clustering_map and composition_map parameters:
# IDENTIFIER_STRING
# K
# INPUT_PICKLED_GRAPH
# ALPHA_VALUES
# VECTOR_FILE_INVARIANT

# probability_composition parameters:
# IDENTIFIER_STRING
# ALPHA_VALUES
# VECTOR_FILE_INVARIANT

# fisher_exact test:
# ATTRIBUTE
# IDENTIFIER_STRING
# K
# ALPHA_VALUES

# calc_ari:
# IDENTIFIER_STRING
# K
# ALPHA_VALUES
# INPUT_PICKLED_GRAPH
# labeling_file

def main():
    # General experimentation pipeline:
    # 1. Pipelines "build_*" to build a relevant graph pickle and edgelist for simulations: in main_pipelines.
    # 2. Simulations to generate vector files for each of the alpha values: with run.sh.
    # 3. Gap, Silhouette, and/or Elbow methods to find K: in main_pipelines.
    # 4. Pipeline "after_vectors" to run clustering Information Access and Spectral Clustering methods
    # and generate plots: in main_pipelines.

    # ========== #

    # 1. Pipelines "build_*":
    if sys.argv[1] == "build_twitch":
        twitch.main()
    elif sys.argv[1] == "build_cosponsorship":
        cosponsorship.main()
    elif sys.argv[1] == "build_dblp":
        dblp.main()

    # 2. Files for simulations can be found in output_files directory.

    # 3. Methods for finding K.
    elif sys.argv[1] == "glowing_gap_statistic":
        raise ValueError(
            "glowing_gap_statistic is not being used for the experiments; please switch to granger_gap_statistic")
        # for alpha_value in ALPHA_VALUES:
        #     X = read.read_in_vectors(VECTOR_FILE_INVARIANT.format(str(alpha_value)[2:]))
        #     X = [X[i] for i in X]
        #     X = np.array(X)
        #
        #     gap, reference_inertia, ondata_inertia = compute_gap(KMeans(), X)
        #
        #     # Plotting here, so that I don't change the code from the original source much.
        #     plt.plot(range(1, MAX_CLUSTERS), reference_inertia,
        #              '-o', label='reference')
        #     plt.plot(range(1, MAX_CLUSTERS), ondata_inertia,
        #              '-o', label='data')
        #     plt.legend()
        #     plt.xlabel('k')
        #     plt.ylabel('log(inertia)')
        #     plt.savefig("output_files/{}_gap_alpha_{}_inertia.png".format(IDENTIFIER_STRING, alpha_value),
        #                 bbox_inches='tight')
        #     plt.close()
        #
        #     plt.plot(range(1, MAX_CLUSTERS), gap, '-o')
        #     plt.title("K vs Gap (alpha = {})".format(alpha_value))
        #     plt.ylabel('gap')
        #     plt.xlabel('k')
        #     plt.savefig("output_files/{}_gap_alpha_{}.png".format(IDENTIFIER_STRING, alpha_value), bbox_inches='tight')
        #     plt.close()
    elif sys.argv[1] == "granger_gap_statistic":
        for alpha_value in ALPHA_VALUES:
            print("\n{}: Gap Statistic for {}".format(IDENTIFIER_STRING, alpha_value))
            X = read.read_in_vectors(VECTOR_FILE_INVARIANT.format(str(alpha_value)[2:]))
            X = [X[i] for i in X]
            X = np.array(X)
            gap_statistic_output = granger_gap_statistic(X, alpha_value)
            print("Optimal number of clusters is {} among {}".format(gap_statistic_output[0], gap_statistic_output[1]))
    elif sys.argv[1] == "silhouette_analysis":
        for alpha_value in ALPHA_VALUES:
            print("\n{}: Silhouette Analysis for {}".format(IDENTIFIER_STRING, alpha_value))
            X = read.read_in_vectors(VECTOR_FILE_INVARIANT.format(str(alpha_value)[2:]))
            X = [X[i] for i in X]
            X = np.array(X)
            silhouette_analysis(X, alpha_value)
    elif sys.argv[1] == "elbow_method":
        for alpha_value in ALPHA_VALUES:
            print("\n{}: Elbow Method for {}".format(IDENTIFIER_STRING, alpha_value))
            vector_file_path = VECTOR_FILE_INVARIANT.format(str(alpha_value)[2:])
            vectors = read.read_in_vectors(vector_file_path)
            elbow_method(vector_file_path, vectors, MIN_CLUSTERS, MAX_CLUSTERS)

    # 4. Pipeline after_vectors for clustering and plotting the relevant graphs.
    elif sys.argv[1] == "after_vectors":
        pipeline_after_vectors()

    # Extra: plot the PDF of the entire dataset.
    elif sys.argv[1] == "dataset_pdf":
        dataset_pdf()

    # Extra: Calculate the adjusted rand index between two clusterings
    elif sys.argv[1] == "calc_ari":
        calculate_ari()
    # Extra: creates a csv file that maps each node to its KMeans cluster (reproducible with random_state=1).
    elif sys.argv[1] == "clustering_map":
        clustering_map()
    # Extra: creates a csv file that maps cluster compositions to clusters.
    elif sys.argv[1] == "composition_map":
        composition_map()
    # Extra: runs the Fisher Exact test on the clusters; hardcoded with K=2.
    elif sys.argv[1] == "fisher_exact":
        fisher_exact()
    # Extra: computes the composition of probabilities in the vector files.
    elif sys.argv[1] == "probability_composition":
        probability_composition()
    return

def calculate_ari():
    create_ari_files()
    spectral_clusters = cp.read_in_clusters("output_files/{}_K{}_spectral_ari.csv".format(IDENTIFIER_STRING, K))
    for alpha_value in ALPHA_VALUES:
        cluster_file = "output_files/{}_K{}_{}_information_access_ari.csv".format(IDENTIFIER_STRING, K,
                                                                                  str(alpha_value)[2:])
        info_clusters = cp.read_in_clusters(cluster_file)
        print(f"alpha: {alpha_value}")
        cp.adj_rand_index(info_clusters, spectral_clusters)
    return


def create_ari_files():
    with open(INPUT_PICKLED_GRAPH, "rb") as file:
        graph = pickle.load(file)

    labeling_file = "output_files/{}_K{}_labeling_file.csv".format(IDENTIFIER_STRING, 2)
    if not os.path.isfile(labeling_file):
        raise FileNotFoundError("labeling_file not found")
    cluster_dict = read_in_clusters(labeling_file)
    for alpha_value in ALPHA_VALUES:
        print("create_ari_files for alpha = {}".format(alpha_value))
        graph = assign_clusters(graph, cluster_dict, alpha_value)

        # save csv of node to cluster to later run ari
        node_to_cluster_filename = "output_files/{}_K{}_{}_information_access_ari.csv".format(IDENTIFIER_STRING, K,
                                                                                      str(alpha_value)[2:])
        read.writeout_clusters(graph, node_to_cluster_filename)

    print("create_ari_files for spectral")
    spectral_labeling_file = "output_files/{}_K{}_labeling_file_spectral.csv".format(IDENTIFIER_STRING, K)
    if not os.path.isfile(spectral_labeling_file):
        raise FileNotFoundError("spectral_labeling_file not found")
    spectral_cluster_dict = read_in_spectral(spectral_labeling_file)
    graph = assign_spectral_clusters(graph, spectral_cluster_dict)
    # save csv of node to cluster to later run ari
    node_to_cluster_filename = "output_files/{}_K{}_spectral_ari.csv".format(IDENTIFIER_STRING, K)
    read.writeout_clusters(graph, node_to_cluster_filename)
    return

# Glowing Gap Statistic code adapted from https://glowingpython.blogspot.com/2019/01/a-visual-introduction-to-gap-statistics.html
def compute_inertia(a, X):
    W = [np.mean(pairwise_distances(X[a == c, :])) for c in np.unique(a)]
    return np.mean(W)

def compute_gap(clustering, data):
    """Main function used in glowing_gap_statistic."""
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    reference = np.random.rand(*data.shape)
    reference_inertia = []
    for k in range(1, MAX_CLUSTERS):
        local_inertia = []
        for _ in range(N_REFS):
            clustering.n_clusters = k
            assignments = clustering.fit_predict(reference)
            local_inertia.append(compute_inertia(assignments, reference))
        reference_inertia.append(np.mean(local_inertia))

    ondata_inertia = []
    for k in range(1, MAX_CLUSTERS):
        clustering.n_clusters = k
        assignments = clustering.fit_predict(data)
        ondata_inertia.append(compute_inertia(assignments, data))

    gap = np.log(reference_inertia) - np.log(ondata_inertia)
    return gap, np.log(reference_inertia), np.log(ondata_inertia)


# Granger Gap Statistic code adapted from https://anaconda.org/milesgranger/gap-statistic/notebook
def granger_gap_statistic(data, alpha_value):
    """
    Notes from the source:
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, MAX_CLUSTERS)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    for gap_index, k in enumerate(range(1, MAX_CLUSTERS)):

        # Holder for reference dispersion results
        refDisps = np.zeros(N_REFS)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(N_REFS):
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)

        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)

    x_data = [i for i in range(1, MAX_CLUSTERS)]
    y_data = [gaps[i - 1] for i in x_data]
    plt.scatter(x_data, y_data)
    plt.plot(x_data, y_data)
    plt.xticks(x_data)

    plt.title("Value of K vs. Gap Statistic\nNumber of references used: {}".format(N_REFS))
    plt.xlabel("Value of K")
    plt.ylabel("Gap Statistic")
    plt.savefig("output_files/{}_gap_alpha_{}.png".format(IDENTIFIER_STRING, alpha_value), bbox_inches='tight')
    plt.close()
    return (gaps.argmax() + 1,
            resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal


# Silhouette Analysis code adapted from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
def silhouette_analysis(X, alpha_value):
    """Main function for silhouette_analysis."""
    for n_clusters in range(2, MAX_CLUSTERS):
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        plt.savefig(
            "output_files/{}_sil_alpha_{}_cluster_{}.png".format(IDENTIFIER_STRING, str(alpha_value)[2:], n_clusters))
        plt.close(fig)
    return


# Elbow_method code adapted from https://towardsdatascience.com/k-means-clustering-with-scikit-learn-6b47a369a83c.
def elbow_method(vector_file, vectors, min_k, max_k):
    """Make elbow graph to choose k hyper-parameter for the clustering methods."""
    X = np.array(list(vectors.values()))

    distortions = []
    for i in range(min_k, max_k):
        print("On k value " + str(i))
        kmeans = KMeans(n_clusters=i, random_state=1).fit(X)
        distortions.append(kmeans.inertia_)
        print(kmeans.inertia_)

    # plot
    print(distortions)
    plt.plot(range(min_k, max_k), distortions, marker='o')
    plt.xticks(range(min_k, max_k))
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    vector_file_name = vector_file[:-4].split("_")
    plt.title("Information Access Clustering Elbow Plot (alpha = 0.{})".format(vector_file_name[-2][1:]))
    plt.savefig("output_files/{}_elbow_{}_{}.png".format(IDENTIFIER_STRING, vector_file_name[-2], vector_file_name[-1]),
                bbox_inches='tight')
    plt.close()
    return


def pipeline_after_vectors():
    """Driver function for the after_vectors pipeline."""
    # Loads the input pickled graph into a local variable graph.
    with open(INPUT_PICKLED_GRAPH, "rb") as file:
        graph = pickle.load(file)

    # Begins Information Access Clustering.
    # Prints the passed string in the terminal and writes it to the report file.
    REPORT_FILE.print("\n================INFORMATION ACCESS==================")

    labeling_file = "output_files/{}_K{}_labeling_file.csv".format(IDENTIFIER_STRING, K)
    if not os.path.isfile(labeling_file):
        # composition_map executes information access clustering and saves a composition_map file.
        clustering_file = composition_map()
        # cluster_labeling uses that file to create a matrix of relabeled clusters.
        cluster_labeling.main(clustering_file, labeling_file)
    cluster_dict = read_in_clusters(labeling_file)

    # For each alpha value, performs the information access clustering and plots the results.
    for alpha_value in ALPHA_VALUES:
        graph = assign_clusters(graph, cluster_dict, alpha_value)
        vector_file_path = VECTOR_FILE_INVARIANT.format(str(alpha_value)[2:])

        REPORT_FILE.print("\n+++++{}+++++\n".format(vector_file_path))
        plot_all_attributes(graph, "information_access", vector_file_path=vector_file_path, alpha_value=alpha_value)

    # Begins Spectral Clustering.
    REPORT_FILE.print("\n================SPECTRAL==================")

    spectral_labeling_file = "output_files/{}_K{}_labeling_file_spectral.csv".format(IDENTIFIER_STRING, K)
    if not os.path.isfile(spectral_labeling_file):
        spectral_clustering_file = spectral_composition()
        cluster_labeling_spectral(spectral_clustering_file, spectral_labeling_file)
    spectral_cluster_dict = read_in_spectral(spectral_labeling_file)
    graph = assign_spectral_clusters(graph, spectral_cluster_dict)
    # print(type(graph))
    # for node in graph.nodes:
    #     print("Sample node:", graph.nodes[node])
    #     break

    plot_all_attributes(graph, "spectral")
    return


def spectral_composition():
    with open(INPUT_PICKLED_GRAPH, "rb") as file:
        graph = pickle.load(file)

    output_filename = "output_files/{}_K{}_composition_map_spectral.csv".format(IDENTIFIER_STRING, K)
    with open(output_filename, 'a') as file:
        fieldnames = ["Cluster {}".format(k) for k in range(K)]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        user_obj_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        print("Composition map for spectral")
        graph = spectral_clustering(graph)

        clusters_total = {i: [] for i in range(K)}
        for node_int in range(len(graph.nodes)):
            j = graph.nodes[node_int]["cluster"]
            clusters_total[j].append(node_int)

        row = [clusters_total[k] for k in range(K)]
        user_obj_writer.writerow(row)
    return output_filename


def cluster_labeling_spectral(spectral_clustering_file, spectral_labeling_file):
    with open(INPUT_PICKLED_GRAPH, "rb") as file:
        graph = pickle.load(file)

    cluster_dict = {}
    with open(spectral_clustering_file, "r") as f:
        lines = csv.reader(f)
        first = True
        for row in lines:
            if first:
                first = False
            else:
                for cluster in range(K):
                    nodes = row[cluster][1:-1].split(", ")
                    for node in nodes:
                        cluster_dict[int(node)] = cluster

    with open(spectral_labeling_file, 'a') as file:
        fieldnames = ["id", "cluster"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        user_obj_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        for node_int in range(len(graph.nodes)):
            row = [node_int, cluster_dict[node_int]]
            user_obj_writer.writerow(row)
    return


def read_in_spectral(spectral_labeling_file):
    """Reads clusters into format: {node: cluster}"""
    cluster_dict = {}
    with open(spectral_labeling_file, "r") as f:
        lines = csv.reader(f)
        first = True
        for row in lines:
            if first:
                first = False
            else:
                cluster_dict[int(row[0])] = int(row[1])
    return cluster_dict


def assign_spectral_clusters(graph, spectral_cluster_dict):
    for node in graph.nodes:
        cluster = int(spectral_cluster_dict[node])
        graph.nodes[node]["cluster"] = cluster
    return graph

def assign_clusters(graph, cluster_dict, alpha):
    # TODO: based on cluster_dict created by read_in_clusters, assign cluster label to each node in graph
    for node in graph.nodes:
        cluster = int(cluster_dict[node][alpha])
        graph.nodes[node]["cluster"] = cluster
    return graph


def read_in_clusters(cluster_label_file):
    '''
    read clusters into format: {node: {alpha: cluster, alpha:cluster…}…}}
    '''
    cluster_dict = {}
    with open(cluster_label_file, "r") as f:
        lines = csv.reader(f)
        first = True
        for row in lines:
            if first:
                first = False
            else:
                node = int(row[0])
                cluster_dict[node] = {}
                for index in range(1, len(row)):
                    alpha = ALPHA_VALUES[index - 1]
                    cluster_dict[node][alpha] = row[index]
    return cluster_dict

def fix_dblp(graph):
    for node in graph.nodes:
        if graph.nodes[node][ATTRIBUTE] == -1:
            graph.nodes[node][ATTRIBUTE] = None
    return graph

def information_access_clustering(vectors, k, graph):
    """Runs Information Access Clustering on graph."""
    X = np.array(list(vectors.values()))
    labels = KMeans(n_clusters=k, random_state=1).fit_predict(X)
    for node in graph.nodes:
        graph.nodes[node]["cluster"] = labels[node]
    # Although the graph itself is mutated by the function above,
    # returns a graph pointer for consistency.
    return graph

def spectral_clustering(graph):
    """Runs Spectral Clustering on graph."""
    if nx.is_directed(graph):
        # nx.Graph is used to make sure the adjacency matrix is symmetric, for that's what spectral clustering accepts.
        temp_graph = nx.Graph()
        for edge in graph.edges:
            temp_graph.add_edge(edge[0], edge[1])
        # Extracts only the necessary attribute values to reduce the space complexity.
        # Hence, it doesn't call largest_connected_component_transform().
        attributes_dict = {}
        for node in graph.nodes:
            try:
                attributes_dict[node] = {ATTRIBUTE: graph.nodes[node][ATTRIBUTE]}
            except:
                continue
        nx.set_node_attributes(temp_graph, attributes_dict)
        graph = temp_graph

    # Adapted from https://stackoverflow.com/questions/23684746/spectral-clustering-using-scikit-learn-on-graph-generated-through-networkx
    node_list = list(graph.nodes())
    # Converts graph to an adj matrix with adj_matrix[i][j] represents weight between node i,j.
    adj_matrix = nx.to_numpy_matrix(graph, nodelist=node_list)

    labels = SpectralClustering(affinity = 'precomputed', assign_labels="discretize",random_state=0,n_clusters=K).fit_predict(adj_matrix)
    for node in node_list:
        graph.nodes[node]["cluster"] = labels[node]
    return graph

def plot_all_attributes(graph, cluster_method, vector_file_path=None, alpha_value=None):
    """Decision-maker function for whether the distribution should use the log of the ATTRIBUTE values."""
    if PLOT_PDF:
        plot_attribute_distributions(graph, cluster_method, vector_file_path=vector_file_path, alpha_value=alpha_value)
    else:
        plot_attribute_bar(graph, cluster_method, vector_file_path=vector_file_path, alpha_value=alpha_value)
    return

def plot_attribute_distributions(graph, cluster_method, vector_file_path=None, alpha_value=None):
    """Plots the distribution of some numerical node ATTRIBUTE for nodes in each cluster."""
    if K < 2:
        raise ValueError("K must be more than 1")

    clusters_total = {cluster: [] for cluster in range(K)}
    no_attribute_dict = {cluster: 0.0 for cluster in range(K)}

    # Processes the data -- nodes' available attribute values -- to be used for the analyses.
    for node in graph.nodes:
        cluster = graph.nodes[node]["cluster"]
        # A node can either have the ATTRIBUTE or not: node["attribute"] or node.
        # If has ATTRIBUTE, it either has a value for ATTRIBUTE or not: node["attribute"] = value or node["attribute"].
        # If it has a value, the value can be either a str type or an int/float type.
        # If it's str type, the string can be a word or a str of a number.
        # All of these are handled by the "try" section,
        # assuming that the attribute value that is unavailable is represented by None.
        try:
            value = float(graph.nodes[node][ATTRIBUTE])
            if PDF_LOG:
                clusters_total[cluster].append(math.log(value, LOG_BASE))
            else:
                clusters_total[cluster].append(value)
        except:
            no_attribute_dict[cluster] += 1

    # Computes and writes cluster sizes, clusters' portions from the total,
    # and the percentages of available nodes in them.
    summarize_clusters(clusters_total, no_attribute_dict)

    plt.figure(figsize=(12, 10))
    color_counter = 0
    for cluster in clusters_total:
        input = [i for i in clusters_total[cluster]]
        for i in input:
            if i < 0:
                print(i)
        # print(input)

        try:
            sns.distplot(input, hist=False, kde=True,
                 kde_kws = {'linewidth': 3},
                 label = str(cluster), norm_hist = True, color=COLOR_PALETTE[color_counter])
        except:
            pass
        color_counter += 1

    # Runs and writes the results of Pairwise Kolmogorov-Smirnov and Kruskal-Wallis tests."""
    kolmogorov_smirnov_test(clusters_total)
    kruskal_wallis_test(clusters_total)

    # Settings for x and y ranges for different experiments
    # if attribute == "views":
    #     plt.xlim(-100000, 200000)
    #     plt.xlim(-100000, 500000)
    #     plt.xlim(-1000000, 2000000)
    # if attribute == "followers_count":
    #     plt.xlim(-100000, 200000)
    #     plt.xlim(-50000, 100000)
    # if attribute == "average_favorite_count":
    #     plt.xlim(-1000, 2000)
    #     plt.xlim(-550, 1000)
    # if attribute == "average_retweet_count":
    #     plt.xlim(-500, 2000)
    #     plt.xlim(-400, 1000)
    # if attribute == "world_system":
    #     plt.ylim(0, 1.5)

    # Depending on whether we're taking log of the values, writes the corresponding x-axis label.
    if PDF_LOG:
        plt.xlabel("log({}) with base {}".format(ATTRIBUTE, LOG_BASE))
    else:
        plt.xlabel(ATTRIBUTE)
    # Writes the y-axis label.
    plt.ylabel("PDF")
    # Depending on the clustering method, writes the corresponding title and saves the plot with a unique name.
    if cluster_method == "information_access":
        vector_file_name_tokens = vector_file_path[:-4].split("_")
        print(vector_file_name_tokens)
        if PDF_LOG:
            plt.title("Density at log({}) for different clusters (alpha = {})".format(ATTRIBUTE, alpha_value))
        else:
            plt.title("Density at {} for different clusters (alpha = {})".format(ATTRIBUTE, alpha_value))
        plt.savefig("output_files/{}_PDF_K{}_{}_{}_{}_vs_{}.png".format(IDENTIFIER_STRING, K, vector_file_name_tokens[-2], vector_file_name_tokens[-1], ATTRIBUTE, cluster_method), bbox_inches='tight')
    elif cluster_method == "spectral":
        if PDF_LOG:
            plt.title("Density at log({}) for different clusters".format(ATTRIBUTE))
        else:
            plt.title("Density at {} for different clusters".format(ATTRIBUTE))
        plt.savefig("output_files/{}_PDF_K{}_{}_vs_{}.png".format(IDENTIFIER_STRING, K, ATTRIBUTE, cluster_method), bbox_inches='tight')
    return

def summarize_clusters(clusters_total, no_attribute_dict):
    """Computes and writes cluster sizes, clusters' portions from the total,
    and the percentages of available nodes in them."""
    # Computes and writes the cluster sizes.
    cluster_sizes = {cluster: len(clusters_total[cluster]) for cluster in range(K)}
    REPORT_FILE.print("\nCluster sizes:" + str(cluster_sizes))

    # Finds the total number of nodes that have available attribute values.
    total_num_of_nodes = 0
    for cluster in cluster_sizes:
        total_num_of_nodes += cluster_sizes[cluster]

    portions = {}

    # For each cluster, writes its portion from the total and the percent of available nodes in it.
    for cluster in cluster_sizes:
        # Finds and writes the portion of the cluster from the total.
        portion = cluster_sizes[cluster] / total_num_of_nodes

        portions[cluster] = portion

        REPORT_FILE.print("\n" + "Portion of {} from the total: {}".format(cluster, portion))
        # Finds and writes the percent available nodes in the cluster.
        available_percent = cluster_sizes[cluster] / (cluster_sizes[cluster] + no_attribute_dict[cluster])
        REPORT_FILE.print("\n" + f"Percent with {ATTRIBUTE} available in cluster {cluster}: {available_percent}")

    # Run chi2 test to see whether there is a relationship between cluster and having ATTRIBUTE data
    num_with_data = np.array(list(cluster_sizes.values()))
    num_without_data = np.array(list(no_attribute_dict.values()))
    r_c_table = np.array((num_with_data, num_without_data))
    # print(r_c_table)
    try:
        g, p, dof, expctd = stats.chi2_contingency(r_c_table)
        REPORT_FILE.print("\n" + f"pvalue from chi2 two-way test of significant relationship between cluster and having {ATTRIBUTE} data: {p}")
    except ValueError as e:
        REPORT_FILE.print(str(e))

    return

def kolmogorov_smirnov_test(clusters_total):
    """Runs and writes the results of Pairwise Kolmogorov-Smirnov test."""
    for i in range(K):
        current_num = K - 1 - i
        for j in range(current_num):
            REPORT_FILE.print("\n{} to {}".format(j, current_num))

            test_output = stats.ks_2samp(clusters_total[j], clusters_total[current_num])
            REPORT_FILE.print("\n" + str(test_output))
    return

def kruskal_wallis_test(clusters_total):
    """Runs and writes the results of Kruskal-Wallis test."""
    arg_list = [clusters_total[i] for i in range(K)]
    REPORT_FILE.print("\nkruskal-wallis, {}-clusters:\n".format(K))

    test_output = stats.kruskal(*arg_list)
    REPORT_FILE.print(str(test_output) + "\n")
    return

def plot_attribute_bar(graph, cluster_method, vector_file_path=None, alpha_value=None):
    """Plots a bar graph for the composition of the attribute in each cluster"""
    # Holder for categorical values of the attribute: when we take a set of it,
    # we can determine the nodes' values without hard-coding them.
    nodes = []
    clusters_total = {cluster: [] for cluster in range(K)}
    no_attribute_dict = {cluster: 0.0 for cluster in range(K)}
    for node in graph.nodes:
        node_cluster = graph.nodes[node]["cluster"]
        try:
            if graph.nodes[node][ATTRIBUTE] is not None:
                # Since the data type is categorical, there is no need to convert to int or float or take log.
                value = graph.nodes[node][ATTRIBUTE]

                # Relabels the binary attribute to the attribute name itself for interpretability.
                if value == 0 or value == "0" or value == "False" or value is False:
                    value = "not {}".format(ATTRIBUTE)
                elif value == 1 or value == "1" or value == "True" or value is True:
                    value = ATTRIBUTE

                clusters_total[node_cluster].append(value)
                nodes.append(value)
        except:
            no_attribute_dict[node_cluster] += 1
            continue

    REPORT_FILE.print(str([("Cluster {}".format(i), len(clusters_total[i])) for i in clusters_total]))

    total_size = len(nodes)
    if total_size == 0:
        raise ValueError("Zero nodes")
    set_of_attr_values = set(nodes)
    list_of_attr_values = sorted(set_of_attr_values)
    x_values = [i for i in range(K)]

    attr_sections = {}
    for attr_value in list_of_attr_values:
        y_values = [clusters_total[a_cluster].count(attr_value)/len(clusters_total[a_cluster]) for a_cluster in x_values]
        attr_sections[attr_value] = y_values
        print(y_values)

    offset = [0 for i in x_values]
    for_legend_values = []
    for_legend_labels = []
    color_counter = 0
    for attr_value in list_of_attr_values:
        bar_container_object = plt.bar(x_values, attr_sections[attr_value], bottom=offset,
                                       color=BAR_GRAPH_COLOR_PALETTE[color_counter])
        for_legend_values.append(bar_container_object[0])
        for_legend_labels.append(attr_value)
        offset = np.add(offset, attr_sections[attr_value]).tolist()
        color_counter += 1

    plt.xlabel('Clusters')
    plt.xticks(x_values)
    plt.ylabel('Probability')
    plt.legend(for_legend_values, for_legend_labels)

    if cluster_method == "information_access":
        vector_file_name_tokens = vector_file_path[:-4].split("_")
        print(vector_file_name_tokens)

        plt.title('Frequency of {} across\n{} clusters\n(alpha = {})'.format(ATTRIBUTE, cluster_method, alpha_value))
        plt.savefig("output_files/{}_BG_K{}_{}_{}_{}_vs_{}.png".format(IDENTIFIER_STRING, K, vector_file_name_tokens[-2],
                                                                       vector_file_name_tokens[-1], ATTRIBUTE, cluster_method),
                    bbox_inches='tight')
    else:
        plt.title('Frequency of {} across\n{} clusters'.format(ATTRIBUTE, cluster_method))
        plt.savefig("output_files/{}_BG_K{}_{}_vs_{}.png".format(IDENTIFIER_STRING, K, ATTRIBUTE, cluster_method), bbox_inches='tight')
    return

def dataset_pdf():
    """Plots the PDF of the entire dataset for ATTRIBUTE."""
    with open(INPUT_PICKLED_GRAPH, "rb") as file:
        graph = pickle.load(file)
        if (IDENTIFIER_STRING == "dblp"):
            graph = fix_dblp(graph)
        nodes = []
        if DATASET_LOG:
            for node in graph.nodes:
                try:
                    nodes.append(math.log(float(graph.nodes[node][ATTRIBUTE]), LOG_BASE))
                except:
                    continue
        else:
            for node in graph.nodes:
                try:
                    nodes.append(float(graph.nodes[node][ATTRIBUTE]))
                except:
                    continue
        print("Attribute available for {} nodes (percentage: {}):".format(len(nodes), len(nodes)/len(graph)), nodes)
        sns.distplot(nodes, hist=False, kde=True,
                     kde_kws={'linewidth': 3},
                     label='dataset', norm_hist=True)

        if DATASET_LOG:
            plt.xlabel("log({}) with base {}".format(ATTRIBUTE, LOG_BASE))
        else:
            plt.xlabel(ATTRIBUTE)
        plt.ylabel("PDF")
        if DATASET_LOG:
            plt.title("Density at log({}) with base {} for different clusters".format(ATTRIBUTE, LOG_BASE))
            plt.savefig("output_files/{}_PDF_full_log_{}.png".format(IDENTIFIER_STRING, ATTRIBUTE), bbox_inches='tight')
        else:
            plt.title("Density at {} for different clusters".format(ATTRIBUTE))
            plt.savefig("output_files/{}_PDF_full_{}.png".format(IDENTIFIER_STRING, ATTRIBUTE), bbox_inches='tight')
        plt.clf()


def clustering_map():
    with open(INPUT_PICKLED_GRAPH, "rb") as file:
        graph = pickle.load(file)

    fieldnames = ["node_int"]
    fieldnames.extend(["cluster at alpha = {}".format(alpha_value) for alpha_value in ALPHA_VALUES])

    rows = [[node_int] for node_int in range(len(graph))]

    for alpha_value in ALPHA_VALUES:
        print("Clustering map for alpha = {}".format(alpha_value))
        vector_file_path = VECTOR_FILE_INVARIANT.format(str(alpha_value)[2:])
        vectors = read.read_in_vectors(vector_file_path)

        graph = information_access_clustering(vectors, K, graph)

        # Can iterate over range(len(graph.nodes)) because the nodes are numbered from 0 to len(graph.nodes).
        # Will iterate over range(len(graph.nodes)) because it makes the ultimate file ordered.
        for node_int in range(len(graph.nodes)):
            rows[node_int].append(graph.nodes[node_int]["cluster"])

    with open("output_files/{}_K{}_cluster_map.csv".format(IDENTIFIER_STRING, K), 'a') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        user_obj_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        for row in rows:
            user_obj_writer.writerow(row)
    return


def composition_map():
    with open(INPUT_PICKLED_GRAPH, "rb") as file:
        graph = pickle.load(file)

    output_filename = "output_files/{}_K{}_composition_map.csv".format(IDENTIFIER_STRING, K)
    with open(output_filename, 'a') as file:
        fieldnames = ["Alpha_Values"]
        fieldnames.extend(["Cluster {}".format(k) for k in range(K)])
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        user_obj_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        for alpha_value in ALPHA_VALUES:
            print("Composition map for alpha = {}".format(alpha_value))
            vector_file_path = VECTOR_FILE_INVARIANT.format(str(alpha_value)[2:])
            vectors = read.read_in_vectors(vector_file_path)

            # Clusters the nodes: the vertices have a "cluster" attribute with a number of the cluster to which they were
            # assigned. If the loop has performed the loop one, updates the "cluster" values for the same graph based
            # on the clustering from the new vector file corresponding to the current alpha value.
            graph = information_access_clustering(vectors, K, graph)

            clusters_total = {i: [] for i in range(K)}
            for node_int in range(len(graph.nodes)):
                j = graph.nodes[node_int]["cluster"]
                clusters_total[j].append(node_int)

            row = [alpha_value]
            row.extend(clusters_total[k] for k in range(K))
            user_obj_writer.writerow(row)
    return output_filename


def fisher_exact():
    with open(INPUT_PICKLED_GRAPH, "rb") as file:
        graph = pickle.load(file)

    with open("output_files/{}_{}_fisher_exact.csv".format(IDENTIFIER_STRING, ATTRIBUTE), 'a') as file:
        fieldnames = ["alpha_value", "p_value", "contingency_table"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        user_obj_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        labeling_file = "output_files/{}_K{}_labeling_file.csv".format(IDENTIFIER_STRING, 2)
        if not os.path.isfile(labeling_file):
            raise FileNotFoundError("labeling_file not found")
        cluster_dict = read_in_clusters(labeling_file)
        for alpha_value in ALPHA_VALUES:
            print("Fisher Exact for alpha = {}".format(alpha_value))
            graph = assign_clusters(graph, cluster_dict, alpha_value)
            fisher_exact_helper(graph, alpha_value, user_obj_writer)

        spectral_labeling_file = "output_files/{}_K{}_labeling_file_spectral.csv".format(IDENTIFIER_STRING, K)
        if not os.path.isfile(spectral_labeling_file):
            raise FileNotFoundError("spectral_labeling_file not found")
        spectral_cluster_dict = read_in_spectral(spectral_labeling_file)
        graph = assign_spectral_clusters(graph, spectral_cluster_dict)
        fisher_exact_helper(graph, "spectral", user_obj_writer)
    return


def fisher_exact_helper(graph, alpha_value, user_obj_writer):
    contingency_table = {ATTRIBUTE: [0, 0], "not-" + ATTRIBUTE: [0, 0]}
    for node_int in range(len(graph.nodes)):
        attribute = graph.nodes[node_int][ATTRIBUTE]
        if attribute == "True" or attribute is True or attribute == 1 or attribute == "1":
            attribute = ATTRIBUTE
        elif attribute == "False" or attribute is False or attribute == 0 or attribute == "0":
            attribute = "not-" + ATTRIBUTE
        cluster = graph.nodes[node_int]["cluster"]
        contingency_table[attribute][cluster] += 1

    odds_ratio, p_value = stats.fisher_exact([contingency_table[i] for i in contingency_table])

    row = [alpha_value, p_value, contingency_table]
    user_obj_writer.writerow(row)
    return


def probability_composition():
    print("Building the composition table")
    histogram_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    holder_dict = {alpha_value: {(i / 10): 0 for i in range(1, 10)} for alpha_value in ALPHA_VALUES}
    smaller_values = {alpha_value: 0 for alpha_value in ALPHA_VALUES}
    all_data = {alpha_value: [] for alpha_value in ALPHA_VALUES}
    for alpha_value in ALPHA_VALUES:
        print("Table component for alpha = {}".format(alpha_value))
        vector_file_path = VECTOR_FILE_INVARIANT.format(str(alpha_value)[2:])
        vectors = read.read_in_vectors(vector_file_path)
        for row in vectors:
            for i in vectors[row]:
                all_data[alpha_value].append(i)
                if i < 0.1:
                    smaller_values[alpha_value] += 1
                else:
                    for j in range(len(histogram_values)):
                        hist_comp_value_i = len(histogram_values) - 1 - j
                        if i >= histogram_values[hist_comp_value_i]:
                            holder_dict[alpha_value][histogram_values[hist_comp_value_i]] += 1
                            break

    histogram_table = [["alpha", "< 0.1", ">= 0.1 && < 0.2", ">= 0.2 && < 0.3", ">= 0.3 && < 0.4", ">= 0.4 && < 0.5",
                        ">= 0.5 && < 0.6", ">= 0.6 && < 0.7", ">= 0.7 && < 0.8", ">= 0.8 && < 0.9", ">= 0.9"]]
    for alpha_value in ALPHA_VALUES:
        histogram_table.append([alpha_value, smaller_values[alpha_value]])
        histogram_table[-1].extend([holder_dict[alpha_value][i] for i in histogram_values])

    with open("output_files/{}_probability_composition.csv".format(IDENTIFIER_STRING), 'a') as file:
        writer = csv.DictWriter(file, fieldnames=histogram_table[0])
        writer.writeheader()
        user_obj_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        for i in range(1, len(histogram_table)):
            user_obj_writer.writerow(histogram_table[i])

    for alpha_value in ALPHA_VALUES:
        print("Building the histogram for alpha = {}".format(alpha_value))
        plt.hist(all_data[alpha_value])
        plt.xlabel("value at p_ij")
        plt.ylabel("count")
        plt.title("Composition of probabilities in information access vectors (alpha = {})".format(alpha_value))
        plt.savefig("output_files/{}_{}_probability_composition.png".format(IDENTIFIER_STRING, str(alpha_value)[2:]),
                    bbox_inches='tight')
        plt.close()
    return

def add_centrality(graph):
    """Adds attributes representing network structure importance metrics to the nodes of graph."""
    deg_centrality = nx.degree_centrality(graph)
    # between_centrality = nx.betweenness_centrality(graph)
    pagerank = nx.pagerank_numpy(graph)
    for node in deg_centrality:
        graph.nodes[node]["degree_centrality"] = deg_centrality[node]
        # graph.nodes[node]["betweeness_centrality"] = between_centrality[node]
        graph.nodes[node]["pagerank"] = pagerank[node]


# ========== [In construction] Consistency methods, recursive and qualitative. ========== #

def consistency(graph):
    """Depending on which consistency method needs to be used, makes the passed graph
    consistent using the base consistency clustering."""

    # with open(PICKLE_INVARIANT.format(str(alpha_value)[2:]), "rb") as file:
    #     graph = pickle.load(file)
    # if vector_file_name[-2][1:] == str(ALPHA_VALUES[0])[2:]:
    #     print("Ground cluster numbering based on alpha = 0.{}".format(vector_file_name[-2][1:]))
    # else:
    #     cluster_consistency(graph, cluster_cases, vector_file_name, attribute)
    #     # alternative_cluster_consistency(graph, vector_file_name, attribute)

    # #     if FIND_CONSISTENCY:
    # #         # take the basis value and cluster
    # #         vector_file = vector_file_invariant.format(str(CONSISTENCY_BASIS_ALPHA)[2:])
    # #         vectors = read.read_in_vectors(vector_file)
    # #         cluster(vectors, K, graph, vector_file, attribute)
    # #         vector_file_name = vector_file[:-4].split("_")
    # #

    # #         with open("output_files/{}_clustered_pickle_K{}_{}_{}_{}_basis".format(IDENTIFIER_STRING, K, vector_file_name[-2],
    # #                                                                          vector_file_name[-1], attribute),
    # #                   'wb') as pickle_file:
    # #             pickle.dump(graph, pickle_file)
    # #
    # #         clusters_total = {cluster: set() for cluster in range(K)}
    # #         for node in graph.nodes:
    # #             cluster = graph.nodes[node]["cluster"]
    # #             clusters_total[cluster].add(node)
    # #
    # #         cluster_correspondence = {}
    # #         # We require attribute values to be determined, since the best-case scenario is for all but one to be 0.
    # #         for cluster in clusters_total:
    # #             composition = {value: 0 for value in ATTRIBUTE_VALUES}
    # #             attribute_unavailable = 0
    # #             for node in clusters_total[cluster]:
    # #                 try:
    # #                     composition[graph.nodes[node][ATTRIBUTE]] += 1
    # #                 except:
    # #                     attribute_unavailable += 1
    # #             cluster_correspondence[cluster] = max(composition, key=composition.get)
    # #         print(cluster_correspondence)
    # #
    # #
    # #         figure out clusters vs attribute
    # #         for each other alpha
    # #             cluster
    # #             convert
    # #             graph
    # #     else:
    # #         for each alpha
    # #             clsuter
    # #             graph
    # #
    # # try:
    # #     with open(INPUT_PICKLED_GRAPH, "rb") as file:
    # #         graph = pickle.load(file)
    # # except:
    # #     graph = INPUT_PICKLED_GRAPH
    #
    #
    # if 1:
    #     # take the basis value and cluster
    #     vector_file = VECTOR_FILE_INVARIANT.format(str(0.6)[2:])
    #     vectors = read.read_in_vectors(vector_file)
    #     graph = spectral_cluster(K, graph)
    #     # cluster(vectors, K, graph, vector_file, ATTRIBUTE)
    #     vector_file_name = vector_file[:-4].split("_")
    #
    #     # with open("output_files/{}_clustered_pickle_K{}_{}_{}_{}_basis".format(IDENTIFIER_STRING, K, vector_file_name[-2],
    #     #                                                                  vector_file_name[-1], attribute),
    #     #           'wb') as pickle_file:
    #     #     pickle.dump(graph, pickle_file)
    #
    #     clusters_total = {cluster: set() for cluster in range(K)}
    #     for node in graph.nodes:
    #         cluster = graph.nodes[node]["cluster"]
    #         clusters_total[cluster].add(node)
    #
    #     cluster_correspondence = {}
    #     uncapt = 0
    #     # We require attribute values to be determined, since the best-case scenario is for all but one to be 0.
    #     for cluster in clusters_total:
    #         composition = {value: 0 for value in ['democrat', 'republican', 'neutral']}
    #         for node in clusters_total[cluster]:
    #             try:
    #                 composition[graph.nodes[node][ATTRIBUTE]] += 1
    #             except:
    #                 uncapt += 1
    #         print(composition)
    #         cluster_correspondence[cluster] = max(composition, key=composition.get)
    #     print(cluster_correspondence)
    #     print(uncapt)

    # if not os.path.isfile(PICKLE_INVARIANT.format(str(GENERAL_ALPHA_VALUES[0])[2:])):
    #     print("\nGenerating clustered pickles for general alpha values.")
    #     for alpha_value in GENERAL_ALPHA_VALUES:
    #         vector_file = vector_file_invariant.format(str(alpha_value)[2:])
    #         vectors = read.read_in_vectors(vector_file)
    #         Transforms the graph to one with a 'cluster' attribute, while saving it as a pickle in clustered_pickles.
    # information_access_clustering(vectors, K, graph, vector_file, attribute)
    # print(type(graph))

    # cluster_cases = consistent_permutation()
    # # Checking the consistency.
    # for alpha_value in ALPHA_VALUES:
    #     with open(PICKLE_INVARIANT.format(str(alpha_value)[2:]), "rb") as file:
    #         graph = pickle.load(file)
    #         compose = {cluster_cases[i][0]: graph.nodes[cluster_cases[i][0]]["cluster"] for i in cluster_cases}
    #         print(alpha_value, compose)
    # case CONCISTENCY == "a":
    #     do blah blah
    return graph

def alternative_cluster_consistency(graph, vector_file_name, attribute):
    print("\ncluster_consistency() for {}".format(vector_file_name))
    path_to_ground_pickle = "output_files/{}_clustered_pickle_K{}_i{}_{}_{}".format(IDENTIFIER_STRING, K, str(ALPHA_VALUES[0])[2:], vector_file_name[-1], attribute)

    if not os.path.isfile(path_to_ground_pickle):
        raise FileNotFoundError("Could not find the ground pickle")

    with open(path_to_ground_pickle, "rb") as file:
        ground_graph = pickle.load(file)
        print(ground_graph.nodes(data=True))

        ground_clusters_total = {cluster: [] for cluster in range(K)}
        for node in ground_graph.nodes:
            cluster = ground_graph.nodes[node]["cluster"]
            ground_clusters_total[cluster].append(node)

        clusters_total = {cluster: [] for cluster in range(K)}
        for node in graph.nodes:
            cluster = graph.nodes[node]["cluster"]
            clusters_total[cluster].append(node)

        # Starts with the biggest cluster; if equal, then in an increasing cluster number.
        sorted_clusters = sorted([(len(clusters_total[cluster]), int(cluster)) for cluster in clusters_total], reverse=True)
        print("Sorted clusters:", sorted_clusters)
        available_clusters = {i for i in range(K)}
        for cluster_tuple in sorted_clusters:
            print("\nCluster tuple:", cluster_tuple)
            print("Available clusters: {}".format(available_clusters), type(available_clusters))
            composition = {cluster: 0 for cluster in range(K)}
            for node in clusters_total[cluster_tuple[1]]:
                try:
                    composition[ground_graph.nodes[node]["cluster"]] += 1
                except:
                    pass
                # print(node, ground_graph.nodes[node]["cluster"])
            print("Composition:", composition, type(composition))
            cluster_tuple_size = len(clusters_total[cluster_tuple[1]])
            current_max = [-1, []]
            for cluster in composition:
                portion = composition[cluster]/cluster_tuple_size
                if portion > current_max[0]:
                    current_max[0] = portion
                    current_max[1] = [cluster]
                elif portion == current_max[0]:
                    current_max[1].append(cluster)
            print("Max portion = {} at {}".format(current_max[0], current_max[1]))
            print(current_max)
            if current_max[0] == 0:
                print("Entered 1")
                raise ValueError("All portions are 0")
            elif len(current_max[1]) == 1:
                print("Entered 2")
                new_cluster = current_max[1][0]
            else:
                print("Entered 3")
                # 1st - first cluster in the equal portion clusters; 0th - length of it.
                min_length = [len(ground_clusters_total[current_max[1][0]]), current_max[1][0]]
                for cluster in current_max[1]:
                    if len(ground_clusters_total[cluster]) < min_length[0]:
                        min_length[0] = len(ground_clusters_total[cluster])
                        min_length[1] = cluster
                new_cluster = min_length[1]

            if new_cluster not in available_clusters:
                print("Process Interrupted: New cluster is not in the available clusters.")
                return -1
            available_clusters.remove(new_cluster)
            if cluster_tuple[1] == new_cluster:
                continue
            else:
                counter = 0
                for node in graph.nodes:
                    if graph.nodes[node]["cluster"] == cluster_tuple[1]:
                        # print("Graph for transformation node cluster type:", type(graph.nodes[node]["cluster"]))
                        graph.nodes[node]["cluster"] = new_cluster
                        # print("Transformed node cluster type (must be integer):", type(graph.nodes[node]["cluster"]))
                        counter += 1
                print("Counter", counter)
    return

def cluster_consistency(graph, cluster_cases, vector_file_name, attribute):
    print("\ncluster_consistency() for {}".format(vector_file_name))
    path_to_ground_pickle = PICKLE_INVARIANT.format(str(ALPHA_VALUES[0])[2:])

    if not os.path.isfile(path_to_ground_pickle):
        raise FileNotFoundError("Could not find the ground pickle")

    with open(path_to_ground_pickle, "rb") as file:
        ground_graph = pickle.load(file)
        print(ground_graph.nodes(data=True))

        clusters_total = {cluster: [] for cluster in range(K)}
        for node in graph.nodes:
            cluster = graph.nodes[node]["cluster"]
            clusters_total[cluster].append(node)
        constituents = [(cluster, len(clusters_total[cluster])) for cluster in range(K)]
        print("Constituent cluster lengths", constituents)

        graph_clusters = [(cluster_cases[cluster][0], graph.nodes[cluster_cases[cluster][0]]["cluster"]) for cluster in cluster_cases]
        print("Graph clusters for the cases:", graph_clusters)

        available_clusters = {i for i in range(K)}
        graph_cluster_set = set()
        # Cluster and vote number
        for ground_cluster in cluster_cases:
            network_id = cluster_cases[ground_cluster][0]
            attribute_value = cluster_cases[ground_cluster][1]
            graph_cluster = graph.nodes[network_id]["cluster"]
            print("Clusters already transformed:", graph_cluster_set)
            print("Case is in", graph_cluster)
            if graph_cluster in graph_cluster_set:
                raise InterruptedError("Some cases are in the same graph cluster")
            graph_cluster_set.add(graph_cluster)
            graph_cluster_attribute_value = graph.nodes[network_id][attribute]
            if attribute_value != graph_cluster_attribute_value:
                raise InterruptedError("Attributes of the nodes don't match")
            if ground_cluster not in available_clusters:
                raise InterruptedError("New cluster is not in the available clusters.")
            available_clusters.remove(ground_cluster)
            if graph_cluster == ground_cluster:
                print("Clusters are already the same at", ground_cluster)
                continue
            else:
                counter = 0
                for node in clusters_total[graph_cluster]:
                    graph.nodes[node]["cluster"] = ground_cluster
                    counter += 1
                print("Counter", counter, "at", ground_cluster)
    return

def check_consistency(cluster, node, composition):
    truth_value = True
    print(composition)
    existing_case_nodes = [composition[i] for i in range(cluster)]
    for alpha_value in ALPHA_VALUES[1:]:
        with open(PICKLE_INVARIANT.format(str(alpha_value)[2:]), "rb") as file:
            current_graph = pickle.load(file)
            variety_set = set()
            for case_node in existing_case_nodes:
                if case_node == -1:
                    continue
                variety_set.add(current_graph.nodes[case_node]["cluster"])
            variety_set.add(current_graph.nodes[node]["cluster"])
            # print(node, variety_set)
        if len(variety_set) != cluster + 1:
            truth_value = False
            break
    return truth_value


def find_consistent_node(cluster, composition, clusters_total):
    truth_value = True
    for i in composition:
        if composition[i] == -1:
            truth_value = False
    if truth_value:
        return True

    nodes = clusters_total[cluster]
    for node in nodes:
        if check_consistency(cluster, node, composition):
            composition[cluster] = node
            if find_consistent_node(cluster + 1, composition, clusters_total):
                return True
    if cluster != 0:
        composition[cluster - 1] = -1
    return False


def consistent_permutation():
    path_to_ground_pickle = PICKLE_INVARIANT.format(str(ALPHA_VALUES[0])[2:])
    with open(path_to_ground_pickle, "rb") as file:
        ground_graph = pickle.load(file)

        clusters_total = {cluster: [] for cluster in range(K)}
        for node in ground_graph.nodes:
            cluster = ground_graph.nodes[node]["cluster"]
            clusters_total[cluster].append(node)

    composition = {cluster: -1 for cluster in range(K)}
    if find_consistent_node(0, composition, clusters_total):
        print("Consistent composition in ground graph:", composition)
        pipeline_cluster_cases = {cluster: (composition[cluster], ground_graph.nodes[composition[cluster]][ATTRIBUTE])
                                  for cluster in composition}
        print(pipeline_cluster_cases)
        return pipeline_cluster_cases
    else:
        raise InterruptedError("Process Interrupted: No consistent permutation found")

if __name__ == "__main__":
    main()
    # with open("output_files/strong-house_pickle", "rb") as file:
    #     graph = pickle.load(file)
    # a_dict = {"0": 0, "1": 0}
    # for node in graph.nodes:
    #     if str(graph.nodes[node]["democrat"]) == "0":
    #         a_dict["0"] += 1
    #     elif str(graph.nodes[node]["democrat"]) == "1":
    #         a_dict["1"] += 1
    # print("democrat", a_dict)
    #     print("len of nodes", len(graph.nodes))
    #     print("len of edges", len(graph.edges))
