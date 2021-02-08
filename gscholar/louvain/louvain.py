import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import sys


def main():
    filename = sys.argv[1]
    print(filename)

    file = open("../datasets/" + filename + ".txt", "r")
    lines = file.readlines()
    graph = nx.Graph()

    counter = 0
    for line in lines:
        if counter != 0:
            s = int(line.split(None, 2)[0])
            t = int(line.split(None, 2)[1])
            graph.add_node(s)
            graph.add_node(t)
            if not graph.has_edge(s, t):
                graph.add_edge(s, t)
            #graph.add_edge(t, s) # for spectral use undirected
        counter +=1
        
    file.close()
    
    n = graph.number_of_nodes()
    print(graph.number_of_nodes())

    # compute the best partition
    #dblp = 6.94, twitch = 5, strong-house = 1.03, gscholar = 11
    partition = community_louvain.best_partition(graph, resolution=5, randomize=None, random_state=None)
    
    zero = 0
    one = 0
    outfilename = filename + "_louvain.txt"
    outfile = open(outfilename, "w")
    for key in partition.keys():
        outfile.write(str(key) + "\t" + str(partition[key]) + "\n")
        if partition[key] >= 2:
            print("error")
        if partition[key] == 1:
            one += 1
        if partition[key] == 0:
            zero += 1
        #print(partition[key])
    outfile.close()
    
    print(zero)
    print(one)
    
    '''
    print(partition)
    for key in partition.keys():
        print(key)
        print(partition[key])
        '''
    
    '''
    # draw the graph
    pos = nx.spring_layout(graph)
    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(graph, pos, partition.keys(), node_size=40,
                           cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(graph, pos, alpha=0.5)
    plt.savefig(filename + ".png")
    #plt.show()
    '''

main()
