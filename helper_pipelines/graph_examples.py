import utils
import numpy as np
import networkx as nx
from copy import *
import matplotlib.pyplot as plt
import random

def depth_two_star(n):
    # n is the number of nodes in the first layer out
    G = nx.OrderedGraph() # Allows multiple edges, to use as proxy for closeness in future

    # center
    G.add_node(0, cluster=None, color=None, activated=False)

    # first layer of spokes
    for node in range(n):
        G.add_node(node, cluster=None, color=None, activated=False)
        G.add_edge(0, node) # These graphs are undirected, so only need to add edge one way

    # second layer of spokes
    count = n
    prev_nodes = deepcopy(G.nodes)
    for node in prev_nodes:
        if node != 0:
            G.add_node(count, cluster=None, color=None, activated=False)
            G.add_edge(node, count)
            count += 1

    return G

def independent_cascade(graph, seeds, alpha, rounds):
    activated = seeds
    for i in seeds:
        graph.node[i]["activated"] = True

    graph_after_each_round = [deepcopy(graph)]
    for round in range(rounds):
        new_active = []
        for k in activated:
            for neighbor in graph.neighbors(k):
                num = random.random()
                if num <= alpha and not graph.node[neighbor]["activated"]:
                    graph.node[neighbor]["activated"] = True
                    new_active.append(neighbor)
        activated = new_active
        print(activated)
        graph_after_each_round.append(deepcopy(graph))

    return graph_after_each_round

def color(graph):
    node_to_color = []
    for node in graph.nodes:
        if graph.node[node]["activated"]:
            node_to_color.append("#1d9bf0")
        else:
            node_to_color.append("#afb5c9")
    return node_to_color

def main():
    graph = depth_two_star(10)
    graphs = independent_cascade(graph, [0], 0.4, 3)
    # pos = nx.spring_layout(graph)

    for graph in graphs:
        color_map = color(graph)
        print(graph)

        # Figure out how to keep order of the nodes consistent in drawing
        nx.draw_kamada_kawai(graph, with_labels=True, node_color = color_map)
        # nx.draw(graph, with_labels=True)
        plt.draw()
        plt.show()

if __name__=="__main__":
    main()
