import random
import networkx as nx

def random_seeds(graph, p):
    n = graph.number_of_nodes()
    seeds = random.sample(range(1, n + 1), p)
    return seeds

def centrality_seeds(graph, p, centrality_type):
    nodes = sorted(graph.nodes(data=True), key=lambda x: x[1][centrality_type], reverse=True)
    seeds = []
    for i in range(p):
        seeds.append(nodes[i][0])
    return seeds

# def connected_component_seeds(graph):
#     largest_cc = max(nx.connected_components(graph), key=len)
#     s = graph.subgraph(largest_cc).copy()
#     return s.nodes(data=False)
