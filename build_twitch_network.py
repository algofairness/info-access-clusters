import csv
import networkx as nx
import pickle
import sys
import build_generic_network as bgn

ATTRIBUTE_CSV_FILE = "data_twitch/ENGB/musae_ENGB_target.csv"
FIELDNAMES = ["id", "days", "mature", "views", "partner", "new_id"]
CSV_EDGELIST_FILENAME = "data_twitch/ENGB/musae_ENGB_edges.csv"


def main():
    twitch()
    return

def twitch():
    G = nx.Graph()
    all_nodes_attributes = {}
    with open(ATTRIBUTE_CSV_FILE, 'r') as attribute_file:
        read_attribute_file = csv.reader(attribute_file)
        next(read_attribute_file)
        for row in read_attribute_file:
            G.add_node(row[-1])
            attr_dict = {}
            for i in range(len(FIELDNAMES)):
                attr_dict[FIELDNAMES[i]] = row[i]
            all_nodes_attributes[row[-1]] = attr_dict
    nx.set_node_attributes(G, all_nodes_attributes)

    with open(CSV_EDGELIST_FILENAME, 'r') as csv_file:
        read_csv = csv.reader(csv_file, delimiter=',')
        next(read_csv)
        for row in read_csv:
            G.add_edge(row[0], row[1])

    G = bgn.largest_connected_component_transform(G)
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    with open("output_files/twitch_pickle", 'wb') as pickle_file:
        pickle.dump(G, pickle_file)

    with open("output_files/twitch_edgelist.txt", 'w') as txt_file:
        num_of_nodes = len(G.nodes)
        directed = 0
        txt_file.write("{}\t{}\n".format(num_of_nodes, directed))
        for edge in G.edges:
            txt_file.write("{}\t{}\n".format(edge[0], edge[1]))

if __name__ == "__main__":
    main()
