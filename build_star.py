import networkx as nx
import pickle
import main_pipelines as mp

def main():
    G = nx.Graph()
    for i in range(1, 21):
        G.add_edge(0, i)

    mp.quick_display(G)

    with open("output_files/star_pickle", 'wb') as pickle_file:
        pickle.dump(G, pickle_file)

    with open("output_files/star_edgelist.txt", 'w') as txt_file:
        num_of_nodes = len(G.nodes)
        directed = 0
        txt_file.write("{}\t{}\n".format(num_of_nodes, directed))
        for edge in G.edges:
            txt_file.write("{}\t{}\n".format(edge[0], edge[1]))
    return

if __name__ == '__main__':
    main()
