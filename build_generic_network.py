import csv
import sys
import networkx as nx
import main_pipelines as mp
import pickle
from networkx.readwrite import json_graph
import json

def populate_network(graph, input_csv_filename):
    '''
    Helper function that gets a generic edgelist at INPUT_CSV_FILENAME as an input in the format from, to
    and populates the graph graph with defined edges (and thus nodes), directed or undirected depending on the graph type.
    :param graph: networkx object to populate; nodes differentiable by node_id, given when scraping data.
    :return: None.
    '''
    with open(input_csv_filename, 'r') as csv_file:
        read_csv = csv.reader(csv_file, delimiter=',')
        next(read_csv)
        for row in read_csv:
            graph.add_edge(int(row[0]), int(row[1]))
    return

def convert_to_nx_graph(graph):
    temp_graph = nx.Graph()
    temp_graph.add_nodes_from(list(graph.nodes(data=True)))
    for edge in graph.edges:
        temp_graph.add_edge(edge[0], edge[1])
    return temp_graph

def largest_connected_component_transform(G):
    '''
    Since convert_to_nx_graph() converts using the exact same naming and attributes of the nodes,
    preserving their consistency, we are allowed to get the largest component of the main graph
    using the largest component of the converted graph.
    :param G: graph to be
    :return:
    '''
    print("Length of G before largest_connected_component(): {}; type of G: {}".format(len(G), type(G)))
    if nx.is_directed(G):
        print("is_directed(G) = True; get nx.strongly_connected_components(G)")
        largest_cc = max(nx.strongly_connected_components(G), key=len)
    else:
        print("is_directed(G) = False; get nx.connected_components(G)")
        largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    print("Length of G after largest_connected_component(): {}; type of G: {}".format(len(G), type(G)))
    return G

def test_plot_attribute_bar():
    # Set K = 3 and IDENTIFIER_STRING = "test" in main_pipelines.py.
    G = nx.Graph()
    G.add_nodes_from(
        [(0, {'cluster': 0, 'value': 'A'}), (1, {'cluster': 0, 'value': 'B'}), (2, {'cluster': 0, 'value': 'A'}),
         (3, {'cluster': 0, 'value': 'A'}), (4, {'cluster': 1, 'value': 'B'}), (5, {'cluster': 1, 'value': 'B'}),
         (6, {'cluster': 1, 'value': 'B'}), (7, {'cluster': 2, 'value': 'A'}), (8, {'cluster': 2, 'value': 'B'}),
         (9, {'cluster': 2, 'value': 'C'}), (10, {'cluster': 2, 'value': 'A'}), (11, {'cluster': 2, 'value': 'B'}),
         (12, {'cluster': 2, 'value': 'C'})])
    mp.plot_all_attributes(G, "some_clustering_method", "test_vectors_ialphavalue_simulationvalue.txt", "value")

def test_case_1_largest_connected_component():
    print("Test Case 1 for largest_connected_component():")
    G = nx.MultiDiGraph()
    true_type = type(G)
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (4, 4), (5, 5)])
    G.add_node(6)

    G = largest_connected_component_transform(G)
    edges = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (4, 4)]
    try:
        if type(G) != true_type:
            print("Test Case 2 Failed (code 0): test_largest_connected_component\n")
            return

        counter = 0
        for edge in G.edges:
            if edge[0] != edges[counter][0] or edge[1] != edges[counter][1]:
                print("Test Case 2 Failed (code 1): test_largest_connected_component\n")
                return
            counter += 1

        print("Test Case 2 Passed: test_largest_connected_component\n")
        return
    except:
        print("Test Case 2 Failed (code 2): test_largest_connected_component\n")
        return

def test_case_2_largest_connected_component():
    print("Test Case 2 for largest_connected_component():")
    G = nx.MultiDiGraph()
    true_type = type(G)
    G.add_edges_from(
        [('0', '1'), ('0', '4'), ('1', '0'), ('4', '0'), ('4', '1'), ('5', '5')])
    G.add_node('2')

    G = largest_connected_component_transform(G)
    edges = [('0', '1'), ('0', '4'), ('1', '0'), ('4', '0'), ('4', '1')]
    try:
        if type(G) != true_type:
            print("Test Case 2 Failed (code 0): test_largest_connected_component\n")
            return

        counter = 0
        for edge in G.edges:
            if edge[0] != edges[counter][0] or edge[1] != edges[counter][1]:
                print("Test Case 2 Failed (code 1): test_largest_connected_component\n")
                return
            counter += 1

        print("Test Case 2 Passed: test_largest_connected_component\n")
        return
    except:
        print("Test Case 2 Failed (code 2): test_largest_connected_component\n")
        return

def test_largest_connected_component():
    test_case_1_largest_connected_component()
    test_case_2_largest_connected_component()

def graph_to_json(graph, output_path):
    data = json_graph.node_link_data(graph)
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, ensure_ascii=False)
    return

def json_to_graph(input_path):
    with open(input_path, 'r') as json_file:
        data = json.load(json_file)
    graph = json_graph.node_link_graph(data)
    return graph

# def set_attributes(graph):
#     '''
#     Helper function that gets a generic csv file of nodes with attributes and populates the graph's nodes by their node_id
#     at NODE_ID with their attributes.
#     :param graph: networkx object to populate; nodes differentiable by node_id, given when scraping data.
#     :return: None.
#     '''
#     all_nodes_attributes = {}
#     with open(ATTRIBUTE_CSV_FILE, 'r') as attribute_file:
#         read_attribute_file = csv.reader(attribute_file)
#         next(read_attribute_file)
#         for row in read_attribute_file:
#             attr_dict = {}
#             for i in range(len(FIELDNAMES)):
#                 attr_dict[FIELDNAMES[i]] = row[i]
#             all_nodes_attributes[int(row[NODE_ID])] = attr_dict
#     nx.set_node_attributes(graph, all_nodes_attributes)
#     return

if __name__ == "__main__":
    # test_largest_connected_component()

    G = nx.MultiDiGraph()
    G = largest_connected_component_transform(G)
    print([edge for edge in G.edges])
