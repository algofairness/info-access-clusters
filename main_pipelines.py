"""
Main experimental pipelines.
"""
import matplotlib
import report_file_object_class as rfo
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys
import math
import os
import csv
import copy
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
import build_generic_network as bgn
import statistics
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from cluster_consistency import cluster_labeling
from karateclub import Role2Vec
from collections import OrderedDict
import cpnet
import numpy as np


# ==== IDENTIFIER: ==== #
# Unique identifier string for each data set.
# Must be either one word (eg. "twitch") or several words using a hyphen
# (eg. "strong-house"). Do NOT use "_" (underscore).
IDENTIFIER_STRING = "twitch"


# ==== PARAMETERS FOR TUNING K = NUMBER OF CLUSTERS: ==== #
MIN_CLUSTERS = 1
MAX_CLUSTERS = 16
N_REFS = 4


# ==== PIPELINE AFTER_VECTORS PARAMETERS: ==== #
K = -1  # Hyperparameter k.
ATTRIBUTE = ""  # Node attribute, about which to create distribution or bar graphs.
INPUT_PICKLED_GRAPH = ""  # Path to a pickled graph with nodes and relevant attributes.
VECTOR_FILE_INVARIANT = ""  # Invariant for the vector txt files for different alpha values.
REPORT_FILE_PATH = ""  # File to which write the results of the clustering experiment.
REPORT_FILE = rfo.ReportFileObject(REPORT_FILE_PATH)  # Instance of the report file (for convenience in writing to it).

# Alpha values for other data sets:
ALPHA_VALUES = []  # Must match the alpha values used for creating the vector files.

PLOT_PDF = 1  # Whether the attribute is for plotting a PDF or not; if 1, plot_attribute_distributions() will be run;
# if 0, plot_attribute_bar().
PDF_LOG = 0  # If PLOT_PDF = 1, when drawing a PDF, should we take log(attribute)?
LOG_BASE = 10  # If PDF_LOG = 1, what is the base for log used in creating a PDF.

# (Optional) parameter for dataset_pdf(), which plots the PDF of the entire dataset.
DATASET_LOG = 1  # If 0, input numbers for PDF are raw; if 1, log of LOG_BASE is taken of them.

# (Optional) Colors used for the graphs.
COLOR_PALETTE = ["#FFC107", "#1E88E5", "#2ECE54", "#EC09D7", "#DDEC4E", "#D81B50", "#CCD85D", "#3701FA", "#D39CA7", "#27EA9F", "#5D5613", "#DC6464"]
BAR_GRAPH_COLOR_PALETTE = ["#BA65A4", "#1A4D68"]


# ==== PARAMETERS FOR CLUSTERING METHODS BEYOND INFORMATION ACCESS AND SPECTRAL: ==== #
IAC_LABELING_FILE = ""
LABELING_FILE = ""
EXPERIMENT = ""

# Repeated fluid communities clustering pipeline hyperparameters:
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Seeds used for random number generation states.

# For core-periphery:
CP_THRESHOLD = 0.5  # Threshold for turning continuous "coreness" measure for each node into binary core/periphery data.


def main():
    # General experimentation pipeline:
    # 1. Pipelines "build_*" to build a relevant graph pickle and edgelist for simulations: in main_pipelines.
    # 2. Simulations to generate vector files for each of the alpha values: with run.sh.
    # 3. Gap, Silhouette, and/or Elbow methods to find K: in main_pipelines.
    # 4. Pipeline "after_vectors" to run clustering Information Access and Spectral Clustering methods
    # and generate plots: in main_pipelines.
    # 5. Run "repeated_fluidc", "role2vec_pipeline", or "core_periphery" clustering methods and
    # use additional methods for deeper analysis.

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
    elif sys.argv[1] == "granger_gap_statistic":
        granger_gap_statistic_wrapper()
    elif sys.argv[1] == "silhouette_analysis":
        silhouette_analysis_wrapper()
    elif sys.argv[1] == "elbow_method":
        elbow_method_wrapper()

    # 4. Pipeline after_vectors for information access and spectral clustering and plotting the relevant graphs.
    elif sys.argv[1] == "after_vectors":
        pipeline_after_vectors()

    # 5. Other clustering methods:
    elif sys.argv[1] == "repeated_fluidc":
        repeated_fluidc()
    elif sys.argv[1] == "fluid_communities":
        fluid_communities()
    elif sys.argv[1] == "role2vec_pipeline":
        role2vec_pipeline()
    elif sys.argv[1] == "core_periphery":
        core_periphery()

    # Additional methods for analysis:
    # Outputs the adjusted rand index scores between clusterings of information access and some other method.
    elif sys.argv[1] == "iac_vs_x_ari":
        iac_vs_x_ari()
    # Computes the mean of adjusted rand index scores across repeated fluid communities clusterings for each alpha value.
    elif sys.argv[1] == "mean_ari":
        mean_ari()
    # Runs the Fisher Exact test on the clusters; hardcoded with K=2.
    elif sys.argv[1] == "fisher_exact":
        fisher_exact()
    # Counts the number of connected components in each cluster, given a clustering in a labeling file.
    elif sys.argv[1] == "count_cc":
        count_cc_wrapper()
    # Given some clustering in the LABELING_FILE, runs statistical analyses for one of DBLP, Co-sponsorship, and Twitch
    # based on its default attributes and settings.
    elif sys.argv[1] == "statistical_analyses":
        statistical_analyses()
    # Plot the PDF of the entire dataset.
    elif sys.argv[1] == "dataset_pdf":
        dataset_pdf()
    # Calculate the adjusted rand index between two clusterings
    elif sys.argv[1] == "calc_ari":
        calculate_ari()
    # Creates a .csv file that maps each node to its KMeans cluster (reproducible with random_state=1).
    elif sys.argv[1] == "clustering_map":
        clustering_map()
    # Creates a .csv file that maps cluster compositions to clusters.
    elif sys.argv[1] == "composition_map":
        composition_map()
    # Computes the composition of probabilities in the information access vector files.
    elif sys.argv[1] == "probability_composition":
        probability_composition()
    # Generates profiles of nodes in .csv (to be used along with edgelist to reconstruct graphs).
    elif sys.argv[1] == "generate_profiles":
        generate_profiles()
    return

# "granger_gap_statistic"
def granger_gap_statistic_wrapper():
    """Wrapper for granger_gap_statistic. Output will be displayed in terminal."""
    for alpha_value in ALPHA_VALUES:
        print("\n{}: Gap Statistic for {}".format(IDENTIFIER_STRING, alpha_value))
        X = read.read_in_vectors(VECTOR_FILE_INVARIANT.format(str(alpha_value)[2:]))
        X = [X[i] for i in X]
        X = np.array(X)
        gap_statistic_output = granger_gap_statistic(X, alpha_value)
        print("Optimal number of clusters is {} among {}".format(gap_statistic_output[0], gap_statistic_output[1]))
    return

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


# "silhouette_analysis"
def silhouette_analysis_wrapper():
    """Wrapper for silhouette_analysis. Output will be displayed in terminal."""
    for alpha_value in ALPHA_VALUES:
        print("\n{}: Silhouette Analysis for {}".format(IDENTIFIER_STRING, alpha_value))
        X = read.read_in_vectors(VECTOR_FILE_INVARIANT.format(str(alpha_value)[2:]))
        X = [X[i] for i in X]
        X = np.array(X)
        silhouette_analysis(X, alpha_value)
    return

# Silhouette Analysis code adapted from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
def silhouette_analysis(X, alpha_value):
    """
    Main function for silhouette_analysis.
    :param X: array of information access vectors.
    :param alpha_value: alpha value.
    :return: None.
    """
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


# "elbow_method"
def elbow_method_wrapper():
    """Wrapper for elbow_method. Output will be displayed in terminal."""
    print("running!")
    for alpha_value in ALPHA_VALUES:
        print("\n{}: Elbow Method for {}".format(IDENTIFIER_STRING, alpha_value))
        vector_file_path = VECTOR_FILE_INVARIANT.format(str(alpha_value)[2:])
        vectors = read.read_in_vectors(vector_file_path)
        elbow_method(vector_file_path, vectors, MIN_CLUSTERS, MAX_CLUSTERS)
    return

# Elbow_method code adapted from https://towardsdatascience.com/k-means-clustering-with-scikit-learn-6b47a369a83c
def elbow_method(vector_file, vectors, min_k, max_k):
    """
    Creates elbow graphs for choosing k (number of clusters) for the clustering methods.
    :param vector_file: vector file path.
    :param vectors: dict of information access vectors by nodes.
    :param min_k: minimum number of clusters to calculate distortion for.
    :param max_k: maximum number of clusters to calculate distortion for.
    :return: None.
    """
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


# "after_vectors"
def pipeline_after_vectors():
    """Driver function for the after_vectors pipeline.
    Runs information access and spectral clustering methods
    and the relevant statistical analysis for the given ATTRIBUTE."""
    # Loads the input pickled graph into a local variable graph.
    with open(INPUT_PICKLED_GRAPH, "rb") as file:
        graph = pickle.load(file)

    # Begins Information Access Clustering.
    # Prints the passed string in the terminal and writes it to the report file.
    REPORT_FILE.print("\n================INFORMATION ACCESS==================")

    labeling_file = "output_files/{}_K{}_labeling_file_iac.csv".format(IDENTIFIER_STRING, K)
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
    spectral_cluster_dict = read_in_generic(spectral_labeling_file)
    graph = assign_generic_clusters(graph, spectral_cluster_dict)
    plot_all_attributes(graph, "spectral")
    return

# "composition_map" (placed here for Top-Down Design)
def composition_map():
    """
    Creates a file that shows the composition (by nodes) of the information access clusters.
    :return: a str path to the output file.
    """
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

def information_access_clustering(vectors, k, graph):
    """
    Runs Information Access Clustering on the graph.
    :param vectors: dict of information access vectors by nodes.
    :param k: number of clusters to enforce.
    :param graph: networkx graph.
    :return: networkx graph, having nodes with a populated "cluster" attribute.
    """
    X = np.array(list(vectors.values()))
    labels = KMeans(n_clusters=k, random_state=1).fit_predict(X)
    for node in graph.nodes:
        graph.nodes[node]["cluster"] = labels[node]
    # Although the graph itself is mutated by the function above,
    # returns a graph pointer for consistency.
    return graph

def read_in_clusters(cluster_label_file):
    """
    Reads a labeling file into format: {node: {alpha: cluster, alpha: cluster…}…}}
    :param cluster_label_file: path to the lableing file.
    :return: clustering dict.
    """
    cluster_dict = {}
    with open(cluster_label_file, "r") as f:
        lines = csv.reader(f)
        first = True
        for row in lines:
            if first:
                alpha_values = [float(alpha_value) for alpha_value in row[1:]]
                first = False
            else:
                node = int(row[0])
                cluster_dict[node] = {}
                for index in range(1, len(row)):
                    alpha = alpha_values[index - 1]
                    cluster_dict[node][alpha] = int(row[index])
    return cluster_dict

def assign_clusters(graph, cluster_dict, alpha):
    """
    From the clustering dict, assigns the cluster label to each node in graph.
    :param graph: networkx graph.
    :param cluster_dict: clustering dict created by read_in_clusters.
    :param alpha: alpha value.
    :return: networkx graph, having nodes with a populated "cluster" attribute.
    """
    for node in graph.nodes:
        cluster = int(cluster_dict[node][alpha])
        graph.nodes[node]["cluster"] = cluster
    return graph

def plot_all_attributes(graph, cluster_method, vector_file_path=None, alpha_value=None):
    """
    Decides which graphs to plot based on whether the ATTRIBUTE is continuous or discrete.
    If continuous, Kolmogorov-Smirnov and Kruskal–Wallis tests are also performed.
    :param graph: networkx graph.
    :param cluster_method: cluster method label (e.g. "iac", "spectral", "role2vec", etc.).
    :param vector_file_path: path to vector files (vector file invariant) if "iac".
    :param alpha_value: alpha value if "iac".
    :return: None.
    """
    if PLOT_PDF:
        plot_attribute_distributions(graph, cluster_method, vector_file_path=vector_file_path, alpha_value=alpha_value, k_clusters=K)
    else:
        plot_attribute_bar(graph, cluster_method, vector_file_path=vector_file_path, alpha_value=alpha_value, k_clusters=K)
    return

def plot_attribute_distributions(graph, cluster_method, vector_file_path=None, alpha_value=None,
                                 identifier_string=IDENTIFIER_STRING, k_clusters=K, attribute=ATTRIBUTE,
                                 pdf_log=PDF_LOG, log_base=LOG_BASE, color_palette=COLOR_PALETTE,
                                 report_file=REPORT_FILE):
    """
    Plots the distribution of the continuous ATTRIBUTE for nodes in each cluster.
    :param graph: networkx graph.
    :param cluster_method: cluster method label (e.g. "iac", "spectral", "role2vec", etc.).
    :param vector_file_path: path to vector files (vector file invariant) if "iac".
    :param alpha_value: alpha value if "iac".
    :param identifier_string: dataset identifier string (e.g. "dblp", "twitch", etc.)
    :param k_clusters: number of clusters to enforce.
    :param attribute: attribute to use for plotting the distribution.
    :param pdf_log: whether to take the log of the attribute value (boolean: True or False).
    :param log_base: if True, the log base.
    :param color_palette: color palette to be used for plotting the graphs.
    :param report_file: report file object to document the statistical test results (Kolmogorov-Smirnov and Kruskal–Wallis).
    :return: None.
    """
    if k_clusters < 2:
        raise ValueError("k_clusters must be more than 1")

    clusters_total = {cluster: [] for cluster in range(k_clusters)}
    no_attribute_dict = {cluster: 0.0 for cluster in range(k_clusters)}

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
            value = float(graph.nodes[node][attribute])
            if pdf_log:
                clusters_total[cluster].append(math.log(value, log_base))
            else:
                clusters_total[cluster].append(value)
        except:
            no_attribute_dict[cluster] += 1

    # Computes and writes cluster sizes, clusters' portions from the total,
    # and the percentages of available nodes in them.
    summarize_clusters(clusters_total, no_attribute_dict, k_clusters, attribute, report_file)

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
                         label=str(cluster), norm_hist=True, color=color_palette[color_counter])
        except:
            pass
        color_counter += 1

    # Runs and writes the results of Pairwise Kolmogorov-Smirnov and Kruskal-Wallis tests."""
    kolmogorov_smirnov_test(clusters_total, k_clusters, report_file)
    kruskal_wallis_test(clusters_total, k_clusters, report_file)

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

    plt.legend()
    # Depending on whether we're taking log of the values, writes the corresponding x-axis label.
    if pdf_log:
        plt.xlabel("log({}) with base {}".format(attribute, log_base))
    else:
        plt.xlabel(attribute)
    # Writes the y-axis label.
    plt.ylabel("PDF")
    # Depending on the clustering method, writes the corresponding title and saves the plot with a unique name.
    if cluster_method == "information_access":
        vector_file_name_tokens = vector_file_path[:-4].split("_")
        print(vector_file_name_tokens)
        if pdf_log:
            plt.title("Density at log({}) for different clusters (alpha = {})".format(attribute, alpha_value))
        else:
            plt.title("Density at {} for different clusters (alpha = {})".format(attribute, alpha_value))
        plt.savefig("output_files/{}_PDF_K{}_{}_{}_{}_vs_{}.png".format(identifier_string, k_clusters,
                                                                        vector_file_name_tokens[-2],
                                                                        vector_file_name_tokens[-1], attribute,
                                                                        cluster_method), bbox_inches='tight')
    else:
        if pdf_log:
            plt.title("Density at log({}) for different clusters".format(attribute))
        else:
            plt.title("Density at {} for different clusters".format(attribute))
        plt.savefig(
            "output_files/{}_PDF_K{}_{}_vs_{}.png".format(identifier_string, k_clusters, attribute, cluster_method),
            bbox_inches='tight')
    plt.close()
    return

def summarize_clusters(clusters_total, no_attribute_dict, k_clusters, attribute, report_file):
    """
    Computes and writes cluster sizes, clusters' portions from the total,
    and the percentages of available nodes in them.
    :param clusters_total: dict {cluster: [attr_value1, attr_value2, ...]}.
    :param no_attribute_dict: dict {cluster: num of nodes with no attribute value}.
    :param k_clusters: number of clusters to enforce.
    :param attribute: node attribute at hand.
    :param report_file: report file object to document the summary.
    :return: None.
    """
    # Computes and writes the cluster sizes.
    cluster_sizes = {cluster: len(clusters_total[cluster]) for cluster in range(k_clusters)}
    report_file.print("\nCluster sizes:" + str(cluster_sizes))

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

        report_file.print("\n" + "Portion of {} from the total: {}".format(cluster, portion))
        # Finds and writes the percent available nodes in the cluster.
        available_percent = cluster_sizes[cluster] / (cluster_sizes[cluster] + no_attribute_dict[cluster])
        report_file.print("\n" + f"Percent with {attribute} available in cluster {cluster}: {available_percent}")

    # Run chi2 test to see whether there is a relationship between cluster and having ATTRIBUTE data:
    num_with_data = np.array(list(cluster_sizes.values()))
    num_without_data = np.array(list(no_attribute_dict.values()))
    r_c_table = np.array((num_with_data, num_without_data))
    try:
        g, p, dof, expctd = stats.chi2_contingency(r_c_table)
        report_file.print(
            "\n" + f"pvalue from chi2 two-way test of significant relationship between cluster and having {attribute} data: {p}")
    except ValueError as e:
        report_file.print(str(e))
    return

def kolmogorov_smirnov_test(clusters_total, k_clusters, report_file):
    """
    Runs and writes the results of Pairwise Kolmogorov-Smirnov test.
    :param clusters_total: dict {cluster: [attr_value1, attr_value2, ...]}.
    :param k_clusters: number of clusters to enforce.
    :param report_file: report file object to document the result.
    :return: None.
    """
    for i in range(k_clusters):
        current_num = k_clusters - 1 - i
        for j in range(current_num):
            report_file.print("\n{} to {}".format(j, current_num))

            test_output = stats.ks_2samp(clusters_total[j], clusters_total[current_num])
            report_file.print("\n" + str(test_output))
    return

def kruskal_wallis_test(clusters_total, k_clusters, report_file):
    """
    Runs and writes the results of Kruskal-Wallis test.
    :param clusters_total: dict {cluster: [attr_value1, attr_value2, ...]}.
    :param k_clusters: number of clusters to enforce.
    :param report_file: report file object to document the result.
    :return: None.
    """
    arg_list = [clusters_total[i] for i in range(k_clusters)]
    report_file.print("\nkruskal-wallis, {}-clusters:\n".format(k_clusters))

    test_output = stats.kruskal(*arg_list)
    report_file.print(str(test_output) + "\n")
    return

def plot_attribute_bar(graph, cluster_method, vector_file_path=None, alpha_value=None,
                       identifier_string=IDENTIFIER_STRING, k_clusters=K, attribute=ATTRIBUTE,
                       color_palette=BAR_GRAPH_COLOR_PALETTE, report_file=REPORT_FILE):
    """
    Plots a bar graph of the cluster composition for the discrete ATTRIBUTE.
    :param graph: networkx graph.
    :param cluster_method: cluster method label (e.g. "iac", "spectral", "role2vec", etc.).
    :param vector_file_path: path to vector files (vector file invariant) if "iac".
    :param alpha_value: alpha value if "iac".
    :param identifier_string: dataset identifier string (e.g. "dblp", "twitch", etc.)
    :param k_clusters: number of clusters to enforce.
    :param attribute: attribute to use for plotting the distribution.
    :param color_palette: color palette to be used for plotting the graphs.
    :param report_file: report file object to document the statistical test results (Kolmogorov-Smirnov and Kruskal–Wallis).
    :return: None.
    """
    # Holder for categorical values of the attribute: when we take a set of it,
    # we can determine the nodes' values without hard-coding them.
    nodes = []
    clusters_total = {cluster: [] for cluster in range(k_clusters)}
    no_attribute_dict = {cluster: 0.0 for cluster in range(k_clusters)}
    for node in graph.nodes:
        node_cluster = graph.nodes[node]["cluster"]
        try:
            if graph.nodes[node][attribute] is not None:
                # Since the data type is categorical, there is no need to convert to int or float or take log.
                value = graph.nodes[node][attribute]

                # Relabels the binary attribute to the attribute name itself for interpretability.
                if value == 0 or value == "0" or value == "False" or value is False:
                    value = "not {}".format(attribute)
                elif value == 1 or value == "1" or value == "True" or value is True:
                    value = attribute

                clusters_total[node_cluster].append(value)
                nodes.append(value)
        except:
            no_attribute_dict[node_cluster] += 1
            continue

    report_file.print(str([("Cluster {}".format(i), len(clusters_total[i])) for i in clusters_total]))

    total_size = len(nodes)
    if total_size == 0:
        raise ValueError("Zero nodes")
    set_of_attr_values = set(nodes)
    list_of_attr_values = sorted(set_of_attr_values)
    x_values = [i for i in range(k_clusters)]

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
                                       color=color_palette[color_counter])
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

        plt.title('Frequency of {} across\n{} clusters\n(alpha = {})'.format(attribute, cluster_method, alpha_value))
        plt.savefig("output_files/{}_BG_K{}_{}_{}_{}_vs_{}.png".format(identifier_string, k_clusters,
                                                                       vector_file_name_tokens[-2],
                                                                       vector_file_name_tokens[-1], attribute,
                                                                       cluster_method),
                    bbox_inches='tight')
    else:
        plt.title('Frequency of {} across\n{} clusters'.format(attribute, cluster_method))
        plt.savefig(
            "output_files/{}_BG_K{}_{}_vs_{}.png".format(identifier_string, k_clusters, attribute, cluster_method),
            bbox_inches='tight')
    plt.close()
    return

def spectral_composition():
    """
    Creates a file that shows the composition (by nodes) of the spectral clusters.
    :return: a str path to the output file.
    """
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

def spectral_clustering(graph):
    """
    Runs Spectral Clustering on the graph.
    :param graph: networkx graph.
    :return: networkx graph, having nodes with a populated "cluster" attribute.
    """
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

def cluster_labeling_spectral(spectral_clustering_file, spectral_labeling_file):
    """
    Creates a labeling file for spectral clustering.
    :param spectral_clustering_file: path to the clustering file from spectral_composition.
    :param spectral_labeling_file: path to the output labeling file.
    :return: None.
    """
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

def read_in_generic(labeling_file):
    """
    Reads clusters into format: {node: cluster}.
    :param labeling_file: labeling file with two columns: node and cluster.
    :return: dict {node: cluster}.
    """
    cluster_dict = {}
    with open(labeling_file, "r") as f:
        lines = csv.reader(f)
        first = True
        for row in lines:
            if first:
                first = False
            else:
                cluster_dict[int(row[0])] = int(row[1])
    return cluster_dict

def assign_generic_clusters(graph, cluster_dict):
    """
    Given a graph, assigns cluster labels to nodes from cluster_dict.
    :param graph: networkx graph.
    :param cluster_dict: dict {node: cluster}.
    :return: networkx graph, having nodes with a populated "cluster" attribute.
    """
    for node in graph.nodes:
        cluster = int(cluster_dict[node])
        graph.nodes[node]["cluster"] = cluster
    return graph


# "repeated_fluidc"
def repeated_fluidc():
    """
    Runs a repeats fluid communities clustering and determines mean and standard deviation
    of the number of connected components in each resulting cluster.
    :return: mean and standard deviation.
    """
    # Access pickled graph:
    with open(INPUT_PICKLED_GRAPH, "rb") as file:
        G = pickle.load(file)

    # Computing labelings:
    seed_to_labeling = {}
    for seed in SEEDS:
        labeling_dict = fluid_communities(save_labeling=False, seed=seed)
        seed_to_labeling[seed] = labeling_dict

    # Documenting:
    clustering_file = "output_files/{}_K{}_composition_map_fluidcr.csv".format(IDENTIFIER_STRING, K)
    with open(clustering_file, 'w') as file:
        # Header:
        fieldnames = ["Seed_Values"]
        fieldnames.extend(["Cluster {}".format(k) for k in range(K)])
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        user_obj_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        for seed in SEEDS:
            print("Composition map for seed value = {}".format(seed))

            clusters = {i: [] for i in range(K)}
            for node in range(len(G)):
                cluster = seed_to_labeling[seed][node]
                clusters[cluster].append(node)

            row = [seed]
            row.extend(clusters[i] for i in range(K))
            user_obj_writer.writerow(row)

    # Relabeling for cluster consistency:
    labeling_file = "output_files/{}_K{}_labeling_file_fluidcr.csv".format(IDENTIFIER_STRING, K)
    cluster_labeling.main(clustering_file, labeling_file)

    # Computation:
    # Form: {node: {seed: cluster, seed: cluster…}…}}
    cluster_dict = read_in_clusters(labeling_file)
    seed_to_counts = {}
    seed_to_sizes = {}
    for seed in SEEDS:
        cluster_dict_by_seed = {node: cluster_dict[node][seed] for node in range(len(G))}
        count_dict, cluster_sizes = count_cc(cluster_dict_by_seed)
        print("Connected component counts:", count_dict)
        seed_to_counts[seed] = count_dict
        seed_to_sizes[seed] = cluster_sizes

    cluster_to_values = {i: [seed_to_counts[seed][i] for seed in SEEDS] for i in range(K)}

    means = {i: statistics.mean(cluster_to_values[i]) for i in range(K)}
    stdevs = {i: statistics.stdev(cluster_to_values[i]) for i in range(K)}

    # Documenting:
    output_filename = "output_files/{}_K{}_fluidcr.txt".format(IDENTIFIER_STRING, K)
    with open(output_filename, mode='w') as file:
        file.write("Means: {}\n".format(means))
        file.write("Standard Deviations: {}".format(stdevs))
    print("Means: {}".format(means))
    print("Standard Deviations: {}".format(stdevs))

    cc_output_filename = "output_files/{}_K{}_cc_fluidcr.csv".format(IDENTIFIER_STRING, K)
    with open(cc_output_filename, mode="w") as file:
        fieldnames = ["seed", "cluster_sizes", "connected_components"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        user_obj_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        for seed in SEEDS:
            row = [seed, seed_to_sizes[seed], seed_to_counts[seed]]
            user_obj_writer.writerow(row)
    return means, stdevs

# "fluid_communities"
def fluid_communities(save_labeling=True, seed=1):
    """
    Single fluid communities clustering pipeline.
    :param save_labeling: whether to save the resulting labeling file or not.
    :param seed: int random number generation state.
    :return: dict {node: cluster}.
    """
    # Access pickled graph:
    with open(INPUT_PICKLED_GRAPH, "rb") as file:
        G = pickle.load(file)

    # Make G undirected to be accepted by asyn_fluidc:
    if nx.is_directed(G):
        G = bgn.convert_to_nx_graph(G)

    # Cluster by asyn_fluidc:
    # Reference: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.asyn_fluid.asyn_fluidc.html#networkx.algorithms.community.asyn_fluid.asyn_fluidc
    clusters = nx.algorithms.community.asyn_fluid.asyn_fluidc(G, k=K, seed=seed)

    # Node-to-cluster dictionary:
    node_to_cluster = {}
    cluster_num = 0
    for cluster in clusters:
        for node in cluster:
            node_to_cluster[node] = cluster_num
        cluster_num += 1

    # Save cluster labels:
    if save_labeling:
        with open(LABELING_FILE, 'w') as file:
            # Header:
            fieldnames = ["node", "asyn_fluidc_cluster_S{}".format(seed)]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            user_obj_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

            # Rows:
            for node in range(len(G)):
                row = [node, node_to_cluster[node]]
                user_obj_writer.writerow(row)
    print("Completed fluid communities clustering with K = {} and S = {}".format(K, seed))
    return node_to_cluster

# "role2vec_pipeline"
def role2vec_pipeline():
    """Pipeline for role2vec clustering, enforcing K."""
    print("role2vec: ", INPUT_PICKLED_GRAPH, IDENTIFIER_STRING)

    with open(INPUT_PICKLED_GRAPH, "rb") as file:
        G = pickle.load(file)

    if nx.is_directed(G):
        G = bgn.convert_to_nx_graph(G)

    # Reference: https://github.com/benedekrozemberczki/karateclub
    role2vec = Role2Vec()
    role2vec.fit(G)
    # Ordered by nodes: [0, 1, 2, ...]
    embedding = role2vec.get_embedding()

    output_filename = "output_files/role2vec_{}_vectors.csv".format(IDENTIFIER_STRING)
    with open(output_filename, mode="w") as output_file:
        user_obj_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        for row in embedding:
            user_obj_writer.writerow(row)


    # Assumption: ordered by nodes as in the data:
    clustering = KMeans(n_clusters=K, random_state=1).fit_predict(embedding)

    clustering_to_labeling_file(G, clustering)
    print("Completed role2vec clustering with K = {}".format(K))
    return

def clustering_to_labeling_file(G, clustering):
    """
    Given a clustering dictionary, creates a labeling file.
    :param G: networkx graph.
    :param clustering: dict {node: cluster}.
    :return: None.
    """
    labeling_filename = "output_files/{}_K{}_labeling_file_role2vec.csv".format(IDENTIFIER_STRING, K)
    with open(labeling_filename, 'w') as file:
        # Header:
        fieldnames = ["node", "role2vec_cluster"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        user_obj_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        # Rows:
        for i in range(len(G)):
            row = [i, clustering[i]]
            user_obj_writer.writerow(row)
    return

# "core_periphery"
def core_periphery():
    """Pipeline for core-periphery clustering, using the CP_THRESHOLD."""
    with open(INPUT_PICKLED_GRAPH, "rb") as file:
        G = pickle.load(file)

    # Reference: https://github.com/skojaku/core-periphery-detection/blob/7d924402caa935e0c2e66fca40457d81afa618a5/cpnet/Rombach.py
    rb = cpnet.Rombach()
    rb.detect(G)
    pair_id = rb.get_pair_id()
    coreness = rb.get_coreness()

    save_cp(pair_id, coreness)
    clustering = make_cp_clustering(coreness)

    # Hardcoded K=2, since binary by threshold:
    filename = "output_files/main_files/{}_K2_labeling_file_cp.csv"
    with open(filename.format(IDENTIFIER_STRING), mode="w") as file:
        # Header:
        fieldnames = ["node", "coreness_binary_{}".format(CP_THRESHOLD)]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        user_obj_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        # Rows:
        for i in range(len(G)):
            row = [i, clustering[i]]
            user_obj_writer.writerow(row)
    return

def save_cp(pair_id, coreness):
    """
    Given pair_id and coreness core_periphery, plots the distribution of coreness and prints the number of
    core-periphery groups in terminal.
    :param pair_id: {node: core-periphery group number} from core_periphery.
    :param coreness: {node: coreness value} from core_periphery.
    :return: None.
    """
    num_of_groups = len(set(list(pair_id.values())))
    input = list(coreness.values())

    sns.distplot(input, hist=False, kde=True,
                 kde_kws={'linewidth': 3},
                 label="dataset", norm_hist=True, color=COLOR_PALETTE[0])
    plt.xlabel("coreness")
    plt.ylabel("PDF")
    plt.title("Density at coreness for {}".format(IDENTIFIER_STRING))
    plt.savefig("output_files/{}_cp_coreness.png".format(IDENTIFIER_STRING), bbox_inches='tight')
    plt.close()
    print("{} num_of_groups = {}".format(IDENTIFIER_STRING, num_of_groups))
    return

def make_cp_clustering(coreness):
    """
    Using the CP_THRESHOLD, turns the continuous coreness into a binary 0/1 clustering.
    :param coreness: {node: coreness value} from core_periphery.
    :return: dict {node: cluster from {0, 1}}
    """
    print("Using core-periphery threshold = {}".format(CP_THRESHOLD))
    clustering = {}
    for node in coreness:
        if coreness[node] <= CP_THRESHOLD:
            clustering[node] = 0
        else:
            clustering[node] = 1
    return clustering


# "iac_vs_x_ari"
def iac_vs_x_ari():
    """Computes and saves the adjusted rand index (ari) scores between information access (IAC_LABELING_FILE)
    and some other (LABELING_FILE) clusterings."""
    # {alpha: ari}
    ari_dict = {}
    # {alpha: OrderedDict({node: cluster})}
    iac_clustering = ordered_read_in_iac()
    # OrderedDict({node: cluster})
    x_clustering = ordered_read_in_x()

    for alpha in ALPHA_VALUES:
        ari_dict[alpha] = cp.return_adj_rand_index(iac_clustering[alpha], x_clustering)

    save_ari(ari_dict)
    return

def ordered_read_in_iac():
    """
    Reads in the information access clustering labeling file into a clustering dictionary, using IAC_LABELING_FILE.
    :return: ordered dict {alpha: OrderedDict({node: cluster})}.
    """
    initial_clustering = read_in_clusters(IAC_LABELING_FILE)
    with open(INPUT_PICKLED_GRAPH, "rb") as file:
        G = pickle.load(file)

    clustering = {alpha: OrderedDict() for alpha in ALPHA_VALUES}
    for node in range(len(G)):
        for alpha in ALPHA_VALUES:
            clustering[alpha][node] = initial_clustering[node][alpha]
    return clustering

def ordered_read_in_x():
    """
    Reads in the other clustering labeling file into a clustering dictionary, using LABELING_FILE.
    :return: ordered dict {alpha: OrderedDict({node: cluster})}.
    """
    initial_clustering = read_in_generic(LABELING_FILE)
    with open(INPUT_PICKLED_GRAPH, "rb") as file:
        G = pickle.load(file)

    clustering = OrderedDict()
    for node in range(len(G)):
        clustering[node] = initial_clustering[node]
    return clustering

def save_ari(ari_dict):
    """
    Saves the given dictionary of ari scores into a .csv file.
    :param ari_dict: dict {alpha: ari score}.
    :return: None.
    """
    x_token = LABELING_FILE.split("_")[-1].replace(".csv", "")

    filename = "output_files/{}_K{}_ari_iac_vs_{}.csv".format(IDENTIFIER_STRING, K, x_token)
    with open(filename, mode="w") as file:
        # Header:
        fieldnames = ["alpha", "ari_score"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        user_obj_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        # Rows:
        for alpha in ALPHA_VALUES:
            row = [alpha, ari_dict[alpha]]
            user_obj_writer.writerow(row)
    print("Saved ari scores for aic vs. {}".format(x_token))
    return

# "mean_ari"
def mean_ari():
    """Computes and documents the mean of adjusted rand index (ari) scores across repeated fluid communities
    clusterings for each alpha value."""
    ari_filename_invariant = "output_files/fluidcr_ari_sa/fluidcr_{}_ari_sa/fluidcrs{}/{}_K{}_ari_iac_vs_fluidcrs{}.csv".format(IDENTIFIER_STRING, "{}", IDENTIFIER_STRING, K, "{}")
    # {seed: {alpha: ari}}
    data = access_ari(ari_filename_invariant)
    # {alpha: mean}
    alpha_to_mean = calculate_means(data)
    document_means(alpha_to_mean)
    return

def access_ari(ari_filename_invariant):
    """
    Reads in ari files across fluid community seeds into a dictionary.
    :param ari_filename_invariant: invariant str path to the fluid community ari file.
    :return: dict {seed: {alpha: ari}}.
    """
    # {seed: {alpha: ari}}
    data = {}
    for seed in SEEDS:
        data[seed] = {}
        ari_filename = ari_filename_invariant.format(seed, seed)
        with open(ari_filename, mode="r") as file:
            next(file)
            for row in file:
                if row[-1] == "\n":
                    row = row[:-1]
                row = row.split(",")
                alpha = float(row[0])
                ari = float(row[1])
                data[seed][alpha] = ari
    return data

def calculate_means(data):
    """
    Calculates the mean ari across fluid community seeds for each alpha value.
    :param data: dict {seed: {alpha: ari}}.
    :return: dict {alpha: mean ari score}.
    """
    # {alpha: [ari_seed_1, ari_seed_2, ...]}
    alpha_to_ari_scores = {alpha: [] for alpha in ALPHA_VALUES}

    for alpha in ALPHA_VALUES:
        for seed in data:
            alpha_to_ari_scores[alpha].append(data[seed][alpha])

    # {alpha: mean}
    alpha_to_mean = {alpha: statistics.mean(alpha_to_ari_scores[alpha]) for alpha in ALPHA_VALUES}
    return alpha_to_mean

def document_means(alpha_to_mean):
    """
    Documents the means of ari values into a .csv file.
    :param alpha_to_mean: dict {alpha: mean ari score}.
    :return: None.
    """
    filename = "output_files/{}_K{}_mean_of_ari_scores.csv"
    with open(filename.format(IDENTIFIER_STRING, K), mode="w") as file:
        # Header:
        fieldnames = ["alpha", "mean_of_ari_scores"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        user_obj_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        # Rows:
        for alpha in ALPHA_VALUES:
            row = [alpha, alpha_to_mean[alpha]]
            user_obj_writer.writerow(row)
    return


# "fisher_exact"
def fisher_exact():
    """Pipeline for running the Fisher Exact test."""
    with open(INPUT_PICKLED_GRAPH, "rb") as file:
        graph = pickle.load(file)

    with open("output_files/{}_{}_fisher_exact.csv".format(IDENTIFIER_STRING, ATTRIBUTE), 'a') as file:
        fieldnames = ["alpha_value", "p_value", "contingency_table"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        user_obj_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        labeling_file = "output_files/{}_K{}_labeling_file_iac.csv".format(IDENTIFIER_STRING, 2)
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
        spectral_cluster_dict = read_in_generic(spectral_labeling_file)
        graph = assign_generic_clusters(graph, spectral_cluster_dict)
        fisher_exact_helper(graph, "spectral", user_obj_writer)
    return

def fisher_exact_helper(graph, label, user_obj_writer, attribute=ATTRIBUTE):
    """
    Helper method for the Fisher Exact test.
    :param graph: networkx graph.
    :param label: label to be used to identify the p-value in the .csv file (e.g. alpha value or "spectral").
    :param user_obj_writer: .csv writer.
    :param attribute: attribute to run the test against.
    :return: None.
    """
    if attribute == "gender":
        contingency_table = {"M": [0, 0], "F": [0, 0]}
    else:
        contingency_table = {attribute: [0, 0], "not-" + attribute: [0, 0]}
    for node_int in range(len(graph.nodes)):
        attr_value = graph.nodes[node_int][attribute]
        if attr_value == "True" or attr_value is True or attr_value == 1 or attr_value == "1":
            attr_value = attribute
        elif attr_value == "False" or attr_value is False or attr_value == 0 or attr_value == "0":
            attr_value = "not-" + attribute
        cluster = graph.nodes[node_int]["cluster"]
        contingency_table[attr_value][cluster] += 1

    odds_ratio, p_value = stats.fisher_exact([contingency_table[i] for i in contingency_table])

    row = [label, p_value, contingency_table]
    user_obj_writer.writerow(row)
    return


# "count_cc"
def count_cc_wrapper():
    """Wrapper for counting the number of connected components in each cluster,
    given the clustering in the LABELING_FILE."""
    mode_token = LABELING_FILE.split("_")[-1].split(".")[0]
    if mode_token == "iac":
        with open("output_files/{}_K{}_cc_{}.csv".format(IDENTIFIER_STRING, K, mode_token), mode="w") as file:
            # Fieldnames and writer:
            fieldnames = ["alpha", "cluster_sizes", "connected_components"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            user_obj_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

            # Counting connected components for each alpha value:
            all_alpha_labeling_dict = read_in_clusters(LABELING_FILE)
            for alpha_value in ALPHA_VALUES:
                print("\nFor alpha = {}".format(alpha_value))
                labeling_dict = {int(node): int(all_alpha_labeling_dict[node][alpha_value]) for node in
                                 all_alpha_labeling_dict}
                count_dict, cluster_sizes = count_cc(labeling_dict)
                print("Connected components: {}".format(count_dict))

                # Documenting:
                row = [alpha_value, cluster_sizes, count_dict]
                user_obj_writer.writerow(row)
    else:
        with open("output_files/{}_K{}_cc_{}.csv".format(IDENTIFIER_STRING, K, mode_token), mode="w") as file:
            # Fieldnames and writer:
            fieldnames = ["clustering_method", "cluster_sizes", "connected_components"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            user_obj_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

            # Counting connected components:
            labeling_dict = read_in_generic(LABELING_FILE)
            count_dict, cluster_sizes = count_cc(labeling_dict)
            print("Connected components: {}".format(count_dict))

            # Documenting:
            row = [mode_token, cluster_sizes, count_dict]
            user_obj_writer.writerow(row)
    return

def count_cc(labeling_dict):
    """
    Counts the number of connected components in each cluster, given the clustering in the LABELING_FILE.
    :param G: graph, from which to take induced subgraphs.
    :param labeling_dict: dict {node: cluster}.
    :param k_clusters: a priori number of clusters used in the LABELING_FILE.
    :return: dict {cluster: number of connected components in its induced subgraph}.
    """
    # Access pickled graph, if G is not specified by argument:
    with open(INPUT_PICKLED_GRAPH, "rb") as file:
        G = pickle.load(file)

    # Iterable of nodes per each cluster as a basis of induced subgraph:
    clusters = {i: [] for i in range(K)}
    for node in range(len(G)):
        clusters[labeling_dict[node]].append(node)

    cluster_sizes = {i: len(clusters[i]) for i in range(len(clusters))}
    print("Cluster sizes: {}".format(cluster_sizes))

    # Counts the number of connected components in each induced subgraph:
    count_dict = {}
    for i in range(K):
        # Induced subgraph by cluster:
        temp_graph = G.subgraph(clusters[i])
        # quick_display(temp_graph)

        # Make temp_graph undirected to be accepted by number_connected_components:
        if nx.is_directed(temp_graph):
            temp_graph = bgn.convert_to_nx_graph(temp_graph)

        # Number of connected components:
        num_of_cc = nx.number_connected_components(temp_graph)
        count_dict[i] = num_of_cc
    return count_dict, cluster_sizes


# "statistical_analyses"
def statistical_analyses():
    """Given some clustering in the LABELING_FILE, runs statistical analyses for one of DBLP, Co-sponsorship, and Twitch
    based on its default attributes and settings."""
    cluster_dict = read_in_generic(LABELING_FILE)
    with open(INPUT_PICKLED_GRAPH, "rb") as file:
        G = pickle.load(file)

    for node in G.nodes:
        G.nodes[node]["cluster"] = cluster_dict[node]

    x_token = LABELING_FILE.split("_")[-1].replace(".csv", "")

    # Invariant for report file:
    report_file_path = "output_files/{}_K{}_{}_output_strings_{}.txt".format(IDENTIFIER_STRING, K, x_token, "{}")

    if IDENTIFIER_STRING == "dblp":
        # Distribution for "citation_count":
        attr_report_file = rfo.ReportFileObject(report_file_path.format("citation_count"))
        plot_attribute_distributions(G, EXPERIMENT, identifier_string=IDENTIFIER_STRING, k_clusters=K,
                                     attribute="citation_count",
                                     pdf_log=0, report_file=attr_report_file)

        # Distribution for "phd_rank":
        attr_report_file = rfo.ReportFileObject(report_file_path.format("phd_rank"))
        plot_attribute_distributions(G, EXPERIMENT, identifier_string=IDENTIFIER_STRING, k_clusters=K,
                                     attribute="phd_rank",
                                     pdf_log=0, report_file=attr_report_file)

        # Distribution for "job_rank":
        attr_report_file = rfo.ReportFileObject(report_file_path.format("job_rank"))
        plot_attribute_distributions(G, EXPERIMENT, identifier_string=IDENTIFIER_STRING, k_clusters=K,
                                     attribute="job_rank",
                                     pdf_log=0, report_file=attr_report_file)

        # Bar graph for "gender":
        attr_report_file = rfo.ReportFileObject(report_file_path.format("gender"))
        plot_attribute_bar(G, EXPERIMENT, identifier_string=IDENTIFIER_STRING, k_clusters=K, attribute="gender",
                           report_file=attr_report_file)
        fisher_exact_modularized(G, EXPERIMENT, identifier_string=IDENTIFIER_STRING, attribute="gender")

    elif IDENTIFIER_STRING == "strong-house":
        # Distribution for "le_score":
        attr_report_file = rfo.ReportFileObject(report_file_path.format("le_score"))
        plot_attribute_distributions(G, EXPERIMENT, identifier_string=IDENTIFIER_STRING, k_clusters=K,
                                     attribute="le_score",
                                     pdf_log=0, report_file=attr_report_file)

        # Bar graph for "democrat":
        attr_report_file = rfo.ReportFileObject(report_file_path.format("democrat"))
        plot_attribute_bar(G, EXPERIMENT, identifier_string=IDENTIFIER_STRING, k_clusters=K, attribute="democrat",
                           report_file=attr_report_file)
        fisher_exact_modularized(G, EXPERIMENT, identifier_string=IDENTIFIER_STRING, attribute="democrat")

    elif IDENTIFIER_STRING == "twitch":
        # Distribution for "views":
        attr_report_file = rfo.ReportFileObject(report_file_path.format("views"))
        plot_attribute_distributions(G, EXPERIMENT, identifier_string=IDENTIFIER_STRING, k_clusters=K,
                                     attribute="views",
                                     pdf_log=1, report_file=attr_report_file)

        # Bar graph for "partner":
        attr_report_file = rfo.ReportFileObject(report_file_path.format("partner"))
        plot_attribute_bar(G, EXPERIMENT, identifier_string=IDENTIFIER_STRING, k_clusters=K, attribute="partner",
                           report_file=attr_report_file)
        fisher_exact_modularized(G, EXPERIMENT, identifier_string=IDENTIFIER_STRING, attribute="partner")
    return


# "dataset_pdf"
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
        print("Attribute available for {} nodes (percentage: {}):".format(len(nodes), len(nodes) / len(graph)), nodes)
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
    return

def fix_dblp(graph):
    """
    Changes the attribute value -1 to None for consistency.
    :param graph: networkx graph.
    :return: networkx graph with modified attribute values for nodes.
    """
    for node in graph.nodes:
        if graph.nodes[node][ATTRIBUTE] == -1:
            graph.nodes[node][ATTRIBUTE] = None
    return graph


# "calc_ari"
def calculate_ari():
    """Previous calculate adjusted rand index method: please instead use the new iac_vs_x_ari method."""
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
    """Helper method for calculate_ari."""
    with open(INPUT_PICKLED_GRAPH, "rb") as file:
        graph = pickle.load(file)

    labeling_file = "output_files/{}_K{}_labeling_file_iac.csv".format(IDENTIFIER_STRING, 2)
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
    spectral_cluster_dict = read_in_generic(spectral_labeling_file)
    graph = assign_generic_clusters(graph, spectral_cluster_dict)
    # save csv of node to cluster to later run ari
    node_to_cluster_filename = "output_files/{}_K{}_spectral_ari.csv".format(IDENTIFIER_STRING, K)
    read.writeout_clusters(graph, node_to_cluster_filename)
    return


# "clustering_map"
def clustering_map():
    """Creates a .csv file that maps each node to its KMeans cluster (reproducible with random_state=1)."""
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


# "probability_composition"
def probability_composition():
    """Computes the composition of probabilities in the information access vector files."""
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
        plt.xlim(xmin=0, xmax=1)
        plt.xlabel("value at p_ij")
        plt.ylabel("count")
        plt.title("Composition of probabilities in information access vectors (alpha = {})".format(alpha_value))
        plt.savefig("output_files/{}_{}_probability_composition.png".format(IDENTIFIER_STRING, str(alpha_value)[2:]),
                    bbox_inches='tight')
        plt.close()
    return


# "generate_profiles"
def generate_profiles(G=None):
    """Generates profiles of nodes in .csv (to be used along with edgelist to reconstruct graphs)."""
    # Access graph through pickle, if G is not passed as an argument:
    if G == None:
        with open(INPUT_PICKLED_GRAPH, "rb") as file:
            G = pickle.load(file)

    # Catch fieldnames except for that which includes "cluster":
    fieldnames = ["node"]
    for fieldname in G.nodes[0]:
        if "cluster" in fieldname:
            continue
        fieldnames.append(fieldname)

    with open("output_files/{}_profiles.csv".format(IDENTIFIER_STRING), 'w') as file:
        # Header:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        user_obj_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        # Rows (iterates of range(len(G)), which corresponds to "node" ids used in edgelist):
        for node in range(len(G)):
            row = [node]
            row.extend([G.nodes[node][attribute] for attribute in fieldnames[1:]])
            user_obj_writer.writerow(row)
    return


# Methods used by external scripts:
def fisher_exact_modularized(G, clustering_method, identifier_string=IDENTIFIER_STRING, attribute=ATTRIBUTE,
                             input_pickled_graph=INPUT_PICKLED_GRAPH):
    """Modularized Fisher Exact test."""
    # Access graph through pickle, if G is not passed as an argument:
    # if G == None:
    #     with open(input_pickled_graph, "rb") as file:
    #         G = pickle.load(file)

    # Fisher Exact:
    with open("output_files/{}_{}_fisher_exact.csv".format(identifier_string, attribute), 'a') as file:
        if os.stat("output_files/{}_{}_fisher_exact.csv".format(identifier_string, attribute)).st_size == 0:
            fieldnames = ["clustering", "p_value", "contingency_table"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
        user_obj_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        # try:
        #     cluster_ex = G.nodes[0]["cluster"]
        # except:
        # cluster_dict = read_in_generic(labeling_file)
        # G = assign_generic_clusters(G, cluster_dict)
        fisher_exact_helper(G, clustering_method, user_obj_writer, attribute=attribute)
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



def quick_display_wrapper_K2(G, labeling_dict):
    """Wrapper for quick_display() with K=2 clusters."""
    # Assigning colors based on binary variable:
    color_map = {i: COLOR_PALETTE[0] if labeling_dict[i] == 1 else COLOR_PALETTE[1] for i in labeling_dict}

    # Displaying the colored graph:
    quick_display(G, color_map=color_map)
    return

def quick_display(G, nodelist=None, color_map=None):
    """
    Quick plotting of the graph.
    :param G: graph.
    :param color_map: must be of form {node: color}.
    :return: None.
    """
    fig = plt.figure()
    # If nodelist not specified, nodelist = sorted nodes iterable:
    if nodelist == None:
        nodelist = [i for i in range(len(G))]
    # If one color specified:
    if color_map == None:
        nx.draw_spring(G, with_labels=True)
    # If one color: color_map == "r":
    elif len(color_map) == 1:
        nx.draw_spring(G, node_color=color_map, with_labels=True)
    # If individual mapping:
    else:
        node_color = [color_map[i] for i in nodelist]
        nx.draw_spring(G, nodelist=nodelist, node_color=node_color, with_labels=True)
    plt.show()
    plt.close()
    return


def cc_cluster_sizes(filename):
    """Reads in a file for cluster sizes."""
    alpha_to_cs = {}
    with open(filename, mode="r") as file:
        next(file)
        for row in file:
            if row[-1] == "\n":
                row = row[:-1]
            row = row.split('"')

            if row[0][-1] == ",":
                alpha = float(row[0][:-1])
            else:
                alpha = float(row[0])

            cs = row[1]
            alpha_to_cs[alpha] = str_to_dict(cs)
    return alpha_to_cs

def str_to_dict(string):
    """Input example: "{0: 214, 1: 224}"."""
    output_dict = {}
    pairs = string[1:-1].split(", ")
    for i in pairs:
        elements = i.split(": ")
        output_dict[int(elements[0])] = int(elements[1])
    return output_dict

def cs_by_search_unnamed(filename, alpha_values):
    """Reads in a file for cluster sizes."""
    alpha_to_cs = {}
    index = 0
    with open(filename, mode="r") as file:
        for row in file:
            if "Cluster sizes:" in row:
                if row[-1] == "\n":
                    row = row[:-1]
                cs = row.replace("Cluster sizes:", "")
                alpha_to_cs[alpha_values[index]] = str_to_dict(cs)
                index += 1
    return alpha_to_cs

def cs_by_binary_output(filename):
    """Reads in a file for cluster sizes."""
    alpha_to_cs = {}
    with open(filename, mode="r") as file:
        for row in file:
            if row[0] == "+":
                alpha = float(row.split("_")[-2].replace("i", "0."))
                alpha_to_cs[alpha] = {}
            elif row[0] == "[":
                if row[-1] == "\n":
                    row = row[:-1]
                row = row[2:-2].split("), (")
                for pair in row:
                    pair = pair.split(", ")
                    pair[0] = pair[0][1:-1].replace("Cluster ", "")
                    cluster = int(pair[0])
                    size = int(pair[1])
                    alpha_to_cs[alpha][cluster] = size
    return alpha_to_cs

def cs_by_continuous_output(filename):
    """Reads in a file for cluster sizes."""
    alpha_to_cs = {}
    with open(filename, mode="r") as file:
        for row in file:
            if "================SPECTRAL==================" in row:
                break
            elif row[0] == "+":
                alpha = float(row.split("_")[-2].replace("i", "0."))
            elif "Cluster sizes:" in row:
                if row[-1] == "\n":
                    row = row[:-1]
                cs = row.replace("Cluster sizes:", "")
                alpha_to_cs[alpha] = str_to_dict(cs)
    return alpha_to_cs


def louvain_preprocess():
    """Preprocesses a .txt clustering file from Louvain method."""
    filename = "output_files/{}_louvain.txt".format(IDENTIFIER_STRING)
    with open(filename, mode="r") as input_file:
        with open(LABELING_FILE, mode="w") as output_file:
            # Header:
            fieldnames = ["node", "louvain_cluster"]
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            user_obj_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

            # Rows:
            for row in input_file:
                if row[-1] == "\n":
                    row = row[:-1]
                row = row.split("\t")
                print(row)
                new_row = [int(row[0]), int(row[1])]
                user_obj_writer.writerow(new_row)
    return

def preprocess_fluidcr(main_labeling_file):
    """Preprocesses a repeated fluid communities clustering file."""
    with open(INPUT_PICKLED_GRAPH, "rb") as file:
        G = pickle.load(file)

    # Form: {node: {seed: cluster, seed: cluster…}…}}
    cluster_dict = read_in_clusters(main_labeling_file)

    for seed in SEEDS:
        cluster_dict_by_seed = {node: cluster_dict[node][seed] for node in range(len(G))}

        filename = "output_files/fluidcr/{}_labeling_files_fluidcr/{}_K{}_labeling_file_fluidcrs{}.csv".format(
            IDENTIFIER_STRING, IDENTIFIER_STRING, K, seed)
        with open(filename, mode="w") as file:
            # Header:
            fieldnames = ["node", "fluidcrs{}".format(seed)]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            user_obj_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

            # Rows:
            for i in range(len(G)):
                row = [i, cluster_dict_by_seed[i]]
                user_obj_writer.writerow(row)
    return


if __name__ == "__main__":
    main()
