import numpy as np
import networkx as nx
import json
import csv
import pickle

def university_to_rank(university):
    '''
    Takes in the name of a university and returns its rank
    (according to the ranking system described in https://advances.sciencemag.org/content/1/1/e1400005)
    '''
    with open("data/dblp/faculty_data - schools.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0] == university:
                return float(row[2])
        print(university)
        return None


# def get_citations(filename):
#     '''
#     Takes in file output by gs_scrape. Returns dictionaries mapping dblp ids to
#     metadata about scholars.
#     '''
#     dict = {} # citation count dictionary
#     gender_dict = {}
#     phd_dict = {}
#     phd_rank_dict = {}
#     job_rank_dict = {}
#     with open(filename) as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         line_count = 0
#         for row in csv_reader:
#             dblp_id = row[-3][1:] # Cuts off space at start of id
#             if int(row[-1]) != -1:
#                 dict[dblp_id] = int(row[-1])
#             else:
#                 dict[dblp_id] = None
#             gender_dict[dblp_id] = row[1]
#             phd_dict[dblp_id] = row[2]
#             phd_rank_dict[dblp_id] = university_to_rank(row[2])
#             job_rank_dict[dblp_id] = university_to_rank(row[3])
#     return dict, gender_dict, phd_dict, phd_rank_dict, job_rank_dict

# for line in reader(file):
def get_citations(filename):
    '''
    Takes in file output by gs_scrape. Returns dictionaries mapping dblp ids to
    metadata about scholars.
    '''
    dict = {} # citation count dictionary
    with open(filename) as csv_file:
        for row in csv.reader(csv_file):
            dblp_id = row[-3][1:] # Cuts off space at start of id
            if int(row[-1]) != -1:
                dict[dblp_id] = int(row[-1])
            else:
                dict[dblp_id] = None
    return dict

def get_all_metadata(faculty_data_file):
    '''
    Takes in faculty data file. Returns dictionaries mapping dblp ids to
    metadata about scholars.
    '''
    gender_dict = {}
    phd_dict = {}
    job_dict = {}
    phd_rank_dict = {}
    job_rank_dict = {}
    with open(faculty_data_file) as csv_file:
        for row in csv.reader(csv_file):
            dblp_id = row[5]
            gender_dict[dblp_id] = row[1]
            phd_dict[dblp_id] = row[2]
            job_dict[dblp_id] = row[3]
            phd_rank_dict[dblp_id] = university_to_rank(row[2])
            job_rank_dict[dblp_id] = university_to_rank(row[3])
    return gender_dict, phd_dict, job_dict, phd_rank_dict, job_rank_dict

def make_network_with_ids(coauthorship_filename, citations_filename):
    '''
    Creates a networkx network based on format of dblp files. Includes node attributes
    based on faculty_data:
    cluster=None, color=None, citation_count, dblp_id, gender, phd (school name), phd_rank (rank of school)

    Coauthorship_filename should indicate a file in adjacency list format, where each
    node is a dblp id.
    Citations_filename should indicate a file that lists each dblp id followed by its
    number of citations.
    '''

    file = open(coauthorship_filename, "r")
    coauthor_lines = file.readlines() # each line is a list of coauthors for one author
    g = nx.Graph() # undirected, no parallel edges

    # get all metadata for nodes:
    gender_to_citations, phd_to_citations, name_to_job, name_to_phd_rank, name_to_job_rank = get_all_metadata("data/dblp/faculty_data - faculty.csv")
    name_to_citations = get_citations(citations_filename)

    # add all nodes  and edges to graph:
    for line in coauthor_lines:
        line = line.split(",")
        if line[-1][-1] == "\n": # eliminates trailing newline
            line[-1] = line[-1][:-1]
        node = line[0]
        if line[0][2:] in name_to_citations: # [2:] fixes disparity with "a/" e.g. in dblp ids
            citations = name_to_citations[line[0][2:]]
            gender = gender_to_citations[line[0][2:]]
            phd = phd_to_citations[line[0][2:]]
            job = name_to_job[line[0][2:]]
            phd_rank = name_to_phd_rank[line[0][2:]]
            job_rank = name_to_job_rank[line[0][2:]]
        else:
            print(line[0][2:])
            citations = None
            gender = None
            phd = None
            phd_rank = None
            job_rank = None
        g.add_node(node, cluster=None, color=None, citation_count = citations, dblp_id = line[0], gender = gender, phd = phd, job = job, phd_rank = phd_rank, job_rank=job_rank)
        for neighbor in line[1:]:
            neighbor_index = neighbor
            if neighbor[2:] in name_to_citations:
                neighbor_citations = name_to_citations[neighbor[2:]]
                neighbor_gender = gender_to_citations[neighbor[2:]]
                neighbor_phd = phd_to_citations[neighbor[2:]]
                neighbor_job = name_to_job[neighbor[2:]]
                neighbor_phd_rank = name_to_phd_rank[neighbor[2:]]
                neighbor_job_rank = name_to_job_rank[neighbor[2:]]
            else:
                print(neighbor[2:])
                neighbor_citations = None
                neighbor_gender = None
                neighbor_phd = None
                neighbor_phd_rank = None
                neighbor_job_rank = None
            g.add_node(neighbor_index, cluster=None, color=None, citation_count = neighbor_citations, dblp_id = neighbor, gender=neighbor_gender, phd=neighbor_phd, job = neighbor_job, phd_rank = neighbor_phd_rank, job_rank=neighbor_job_rank)
            g.add_edge(node, neighbor_index)
    return g

def add_centrality(graph):
    '''
    Adds attributes representing network structure importance metrics to the nodes of graph.
    '''
    deg_centrality = nx.degree_centrality(graph)
    between_centrality = nx.betweenness_centrality(graph)
    pagerank = nx.pagerank_numpy(graph)
    for node in deg_centrality:
        graph.nodes[node]["degree_centrality"] = deg_centrality[node]
        graph.nodes[node]["betweeness_centrality"] = between_centrality[node]
        graph.nodes[node]["pagerank"] = pagerank[node]

def write_graph(graph, seeds, filename, include_seeds):
    '''
    Writes out graph into a file formatted as an edgelist so that it can be read into
    the C++ code to create vectors.

    seeds should be a list of the node ids for any seed nodes if include_seeds= True
    filename is the file to write the edgelist to
    include_seeds should be False if the vectors are going to be created using all
    the nodes as seeds, and True if the vectors will only use a subset of nodes as seeds.
    '''
    nx.write_edgelist(graph, filename, data=False)
    insert_str = str(graph.number_of_nodes()) + "\t 0"
    f = open(filename, 'r'); s = f.read(); f.close()
    l = s.splitlines(); l.insert(0, insert_str); s = '\n'.join(l)
    f = open(filename, 'w'); f.write(s + "\n"); f.close();

    if (include_seeds):
        with open(filename, "a") as f:
            f.write("s\t")
            for seed in seeds:
                f.write(str(seed) + "\t")

def main():
    '''
    Runs the first half of the basic pipeline on only the largest connected component,
    until vectors need to be generated in C++
    '''
    graph = make_network_with_ids("data/dblp/october_coauthorship_dblp_ids.txt", "data/dblp/dblp_id_citations")
    print(f"number of nodes in whole graph: {graph.number_of_nodes()}")
    print(f"number of edges in whole graph: {graph.number_of_edges()}")
    largest_cc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest_cc).copy()
    print(f"number of nodes in sub graph: {graph.number_of_nodes()}")
    print(f"number of edges in sub graph: {graph.number_of_edges()}")
    add_centrality(graph)
    graph = nx.convert_node_labels_to_integers(graph) # replaces dblp id label with integer label. First label is 0.
    with open("output_files/dblp_pickle", "wb") as f:
        pickle.dump(graph, f)
    write_graph(graph, [], "output_files/dblp_edgelist.txt", False)

if __name__ == '__main__':
    main()
