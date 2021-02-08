"""Pipeline for building the triangular cliques graph."""
import main_pipelines as mp
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from random import seed
from random import random
from scipy.stats import bernoulli
import build_generic_network as bgn

import community as community_louvain
import matplotlib.cm as cm

INPUT_PICKLED_GRAPH = "output_files/exp_small-alpha-randcl/randcl_pickle"


def main():
    G = random_cliques_graph()
    print("Length of G: {}".format(len(G)))
    produce_edgelist(G, "randcl")
    return


def cliques_graph():
    # Construct:
    G = nx.Graph()
    for i in range(21):
        G.add_node(i)

    for i in range(6):
        for j in range(i + 1, 7):
            G.add_edge(i, j)

    for i in range(7, 13):
        for j in range(i + 1, 14):
            G.add_edge(i, j)

    for i in range(14, 20):
        for j in range(i + 1, 21):
            G.add_edge(i, j)

    G.add_edge(6, 13)
    G.add_edge(13, 20)

    # Display:
    fig = plt.figure()
    nx.draw_spring(G, with_labels=True)
    plt.show()
    plt.close()

    # Pickle:
    with open("output_files/cliques_pickle", 'wb') as pickle_file:
        pickle.dump(G, pickle_file)
    return G


def random_cliques_graph():
    # Construct:
    G = nx.Graph()
    for i in range(300):
        G.add_node(i)

    intra_random_states = [0, 1, 2]
    for k in range(len(intra_random_states)):
        base = k * 100
        trials = bernoulli.rvs(0.5, size=4950, random_state=intra_random_states[k])
        index = 0
        for i in range(base, base + 99):
            for j in range(i + 1, base + 100):
                if trials[index]:
                    G.add_edge(i, j)
                index += 1

    num_of_intra_edges = len(G.edges)
    print("Expectation(|intra-community edges|) = ~7425:", num_of_intra_edges)

    inter_random_states = [3, 4, 5]
    for k in range(len(inter_random_states) - 1):
        for m in range(k + 1, len(inter_random_states)):
            source_base = k * 100
            target_base = m * 100
            trials = bernoulli.rvs(0.005, size=10000, random_state=inter_random_states[k])
            index = 0
            for i in range(source_base, source_base + 100):
                for j in range(target_base, target_base + 100):
                    if trials[index]:
                        G.add_edge(i, j)
                    index += 1

    num_of_inter_edges = len(G.edges) - num_of_intra_edges
    print("Expectation(|inter-community edges|) = ~150:", num_of_inter_edges)

    # Take the largest connected component:
    G = bgn.largest_connected_component_transform(G)

    # Display:
    fig = plt.figure()
    nx.draw_spring(G, with_labels=True)
    plt.show()
    plt.close()

    # Pickle:
    with open("output_files/randcl_pickle", 'wb') as pickle_file:
        pickle.dump(G, pickle_file)
    return G


def produce_edgelist(G, keyword):
    with open("output_files/{}_edgelist.txt".format(keyword), 'w') as txt_file:
        num_of_nodes = len(G.nodes)
        directed = 0
        txt_file.write("{}\t{}\n".format(num_of_nodes, directed))
        for edge in G.edges:
            txt_file.write("{}\t{}\n".format(edge[0], edge[1]))
    return


def equality_test():
    output = []
    with open("output_files/exp_small-alpha-randcl/randcl_K3_composition_map.csv", mode='r') as file:
        next(file)
        for row in file:
            row = row.split(",")
            new_row = [row[0]]
            print(row[0], row[1])
            # error = 0
            # cluster_nodes = {i - 1: set(row[i]) for i in range(1, 4)}
            # print(cluster_nodes)
            # for i in range(1, 4):
            #     for node in range((i-1) * 100, i * 100)):
            #         if node not in cluster_nodes[i-1]:
            #             error

if __name__ == "__main__":
    main()
