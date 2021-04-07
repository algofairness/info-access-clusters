# import utils
import numpy as np
import networkx as nx
import json
import time
from progress.bar import Bar
import csv


def create_star_matrix(n, directed):
    X = []
    first_row = [0]
    for node in range(n - 1):
        first_row.append(1)
    X.append(first_row)
    for node in range(n - 1):
        node_row = [1]
        for neighbor in range(n - 1):
            node_row.append(0)
        X.append(node_row)
    return np.array(X)


def make_networkx(c_style_filename):
    file = open(c_style_filename, "r")
    lines = file.readlines()
    first_line = lines[0].split("\t")
    if (first_line[1] == "0"):
        G = nx.MultiGraph()  # Allows multiple edges, to use as proxy for closeness in future
    else:
        G = nx.MultiDiGraph()

    for line in lines:
        line = line.split("\t")
        node = int(line[0])
        neighbor = int(line[1])
        G.add_node(node, cluster=None, color=None)
        G.add_node(neighbor, cluster=None, color=None)
        G.add_edge(node, neighbor)

    return G


def make_network_with_citations(coauthorship_filename, citations_filename):
    file = open(coauthorship_filename, "r")
    coauthor_lines = file.readlines()
    g = nx.Graph()

    name_to_citations, gender_to_citations, phd_to_citations, name_to_phd_rank = get_citations(citations_filename)
    # print(gender_to_citations)

    person_to_index = {}
    index = 0
    for index2, line in enumerate(coauthor_lines):
        cpy = line.split(",")
        if cpy[0][-1] == "\n":
            cpy[0] = cpy[0][:-1]
        if cpy[0] not in person_to_index:
            person_to_index[cpy[0]] = index
            index += 1

    counter = 0
    for line in coauthor_lines:
        counter += 1
        line = line.split(",")
        if line[-1][-1] == "\n":
            line[-1] = line[-1][:-1]
        node = person_to_index[line[0]]
        # citations = name_to_citations[eliminate_middle_inits(line[0])]
        if line[0][2:] in name_to_citations:
            citations = name_to_citations[line[0][2:]]
            gender = gender_to_citations[line[0][2:]]
            phd = phd_to_citations[line[0][2:]]
            phd_rank = name_to_phd_rank[line[0][2:]]
        else:
            citations = -1
            gender = 'not found'
            # print('not found')
        g.add_node(node, cluster=None, color=None, citation_count=citations, dblp_id=line[0], gender=gender, phd=phd,
                   phd_rank=phd_rank)
        for neighbor in line[1:]:
            neighbor_index = person_to_index[neighbor]
            # neighbor_citations = name_to_citations[eliminate_middle_inits(neighbor)]
            if neighbor[2:] in name_to_citations:
                neighbor_citations = name_to_citations[neighbor[2:]]
                neighbor_gender = gender_to_citations[neighbor[2:]]
                neighbor_phd = phd_to_citations[neighbor[2:]]
                neighbor_phd_rank = name_to_phd_rank[neighbor[2:]]
            else:
                neighbor_citations = -1
                neighbor_gender = 'not found'
                # print ("not found")
                neighbor_phd = "not found"
            g.add_node(neighbor_index, cluster=None, color=None, citation_count=neighbor_citations, dblp_id=neighbor,
                       gender=neighbor_gender, phd=neighbor_phd, phd_rank=neighbor_phd_rank)
            g.add_edge(node, neighbor_index)
    return g


def eliminate_middle_inits(name):
    new_name = name
    for index, char in enumerate(name[:-1]):
        if char == ".":
            new_name = name[:index - 2] + name[index + 1:]
    return new_name


def university_to_rank(university):
    '''
    Takes in the name of a university and returns its rank
    (according to the ranking system described in https://advances.sciencemag.org/content/1/1/e1400005)
    '''
    with open("data/dblp/faculty_data - schools.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0] == university[1:]:
                return float(row[2])
        return -1


def get_citations(filename):
    '''
    Takes in file output by gs_scrape. Returns dictionaries mapping dblp ids to
    metadata about scholars.
    '''
    dict = {}  # citation count dictionary
    gender_dict = {}
    phd_dict = {}
    phd_rank_dict = {}
    job_rank_dict = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            dblp_id = row[-3][1:]  # Cuts off space at start of id
            dict[dblp_id] = int(row[-1])
            gender_dict[dblp_id] = row[1]
            phd_dict[dblp_id] = row[2]
            phd_rank_dict[dblp_id] = university_to_rank(row[2])
            job_rank_dict[dblp_id] = university_to_rank(row[3])
    return dict, gender_dict, phd_dict, phd_rank_dict, job_rank_dict


def read_data(filename):
    # Reads data in from adjancency list text file to adjancency list dict
    file = open(filename, "r")
    authors = file.readlines()
    graph_for_igraph = []
    logical_graph = {}

    for line in authors:
        line = line[:-1]
        line = line.split(",")
        author = line[0]
        logical_graph[author] = []
        for j in line[1:]:
            logical_graph[author].append(j)
            graph_for_igraph.append({'from': author, 'to': j})

    person_to_index = {}
    for index, node in enumerate(logical_graph):
        person_to_index[node] = index

    graph = {}
    for node in logical_graph:
        graph[person_to_index[node]] = []
        for neighbor in logical_graph[node]:
            graph[person_to_index[node]].append(person_to_index[neighbor])

    return graph, graph_for_igraph


def write_graph_as_edges(graph, filename):
    # Writes adjancency list graph to filename
    file = open(filename, "w")
    n = len(graph)
    file.write(str(n) + "\t" + str(1) + "\n")
    person_to_index = {}
    for index, node in enumerate(graph):
        person_to_index[node] = index
    for index, node in enumerate(graph):
        for neighbor in graph[node]:
            file.write(str(index) + "\t" + str(person_to_index[neighbor]) + "\n")
    file.close()


# def write_graph_with_seeds(graph, filename, seeds):
#     # Writes adjancency list graph to filename
#     print(seeds)
#     file = open(filename, "w")
#     n = len(graph)
#     file.write(str(n) + "\t" + str(1) + "\n")
#     person_to_index = {}
#     for index, node in enumerate(graph):
#         person_to_index[node] = index
#     print(enumerate(graph))
#     for index, node in enumerate(graph):
#         for neighbor in graph[node]:
#             file.write(str(index) + "\t" + str(person_to_index[neighbor]) + "\n")
#     file.write("s\t")
#     for seed in seeds:
#         file.write(str(person_to_index[graph.node[seed]])+"\t")
#     file.close()

# help from https://stackoverflow.com/questions/4454298/prepend-a-line-to-an-existing-file-in-python
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
    f = open(filename, 'r');
    s = f.read();
    f.close()
    l = s.splitlines();
    l.insert(0, insert_str);
    s = '\n'.join(l)
    f = open(filename, 'w');
    f.write(s + "\n");
    f.close();

    if (include_seeds):
        with open(filename, "a") as f:
            f.write("s\t")
            for seed in seeds:
                f.write(str(seed) + "\t")


def create_star():
    # Creates a 20-pointed star graph
    graph = {}
    points = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    graph = {point: [0] for point in points}
    graph[0] = points
    print(graph)
    return graph


def create_networkx_star(n, directed):
    # n is number of nodes in the star
    if (not directed):
        G = nx.MultiGraph()  # Allows multiple edges, to use as proxy for closeness in future
    else:
        G = nx.MultiDiGraph()
    G.add_node(0, cluster=None, color=None)
    for node in range(n - 1):
        G.add_node(node, cluster=None, color=None)
        G.add_edge(0, node)
        G.add_edge(node, 0)
    return G


def read_in_vectors(filename):
    '''
    Takes in a filepath with probability vectors and returns a dict of prob vectors
    '''
    file = open(filename, "r")
    nodes = file.readlines()
    vectors = {}
    for index, line in enumerate(nodes):
        line = line.split(",")
        # print("the length of the line is ", len(line))
        node = index
        vectors[node] = []
        for prob in line[:-1]:
            # print(prob)
            vectors[node].append(float(prob))
    return vectors


def read_in_seed_vectors(filename):
    '''
    Takes in a filepath with probability vectors and returns a dict of prob vectors
    '''
    file = open(filename, "r")
    nodes = file.readlines()
    vectors = {}
    for index, line in enumerate(nodes):
        if index != 0:  # deals with first line that lists seeds
            line = line.split(",")
            # print("the length of the line is ", len(line))
            node = index
            vectors[node] = []
            for prob in line[:-1]:
                # print(prob)
                vectors[node].append(float(prob))
    return vectors


def create_hardcoded_stars(alpha, n):
    # n is number of spokes, (it does not include the center)
    vectors = []
    center_vector = [1]
    for i in range(n):
        center_vector.append(alpha)
    vectors.append(center_vector)
    for i in range(n):
        ith_vector = [alpha]
        for j in range(n):
            if (i == j):
                ith_vector.append(1)
            else:
                ith_vector.append(alpha ** 2)
        vectors.append(ith_vector)
    return np.array(vectors)


# def create_mini_gs_graph(id, levels_out):
#     # This function converts the google scholar file to a file that can be run through c code
#     start_time = time.time()
#     graph = {id:[]}
#     paper_to_authors = {}
#     with open(in_filename, 'r') as fobj:
#         lines = fobj.readlines()
#         for line in lines:
#             if line == "\n":
#                 pass
#             line = line.encode().decode('utf-8-sig')
#             data = json.loads(line)
#             print (type(data))
#             print ("processing" + data['name'] + " in first for loop after" + str(time.time() - start_time) + "seconds")
#             for paper in data['paper']:
#                 if paper['paper_id'] in paper_to_authors:
#                     paper_to_authors[paper['paper_id']].append(data['google_id'])
#                 else:
#                     paper_to_authors[paper['paper_id']] = [data['google_id']]
#
#     # Attempt to clear up some memory
#     del lines
#     del line
#     del data
#
#     for paper in paper_to_authors:
#         print ("processing " + paper + "in second for loop after" + str(time.time() - start_time) + "seconds")
#         in_network = False
#         for author in paper_to_authors[paper]:
#             if author in graph:
#                 in_network = True
#         if in_network:
#             for author in paper_to_authors[paper]:
#                 if author in graph:
#                     graph[author] += paper_to_authors[paper]
#                 else:
#                     graph[author] = paper_to_authors[paper]
#                 # print(graph[author])
#                 # coauthors = []
#                 # for paper in datum['paper']:
#                 #     for author in paper['author_list']:
#                 #         coauthors += ''
#                 # graph[datum['google_id']] = coauthors
#     return graph


def gs_to_c_style(in_filename, out_filename):
    # This function converts the google scholar file to a file that can be run through c code
    start_time = time.time()
    graph = {}
    paper_to_authors = {}
    author_to_index = {}
    index = -1
    with open(in_filename, 'r') as fobj:
        lines = fobj.readlines()
        n = len(lines)
        for line in lines:
            if line == "\n":
                pass
            line = line.encode().decode('utf-8-sig')
            data = json.loads(line)
            print(type(data))
            print("processing" + data['name'] + " in first for loop after" + str(time.time() - start_time) + "seconds")
            index += 1
            for paper in data['paper']:
                if paper['title'] in paper_to_authors:
                    paper_to_authors[paper['title']].append(index)
                else:
                    paper_to_authors[paper['title']] = [index]

        # Attempt to clear up some memory
        del lines
        del line
        del data

        # write out to file
    file = open(out_filename, "w")
    bar = Bar('Processing')
    file.write(str(n) + "\t1\n")
    count = 0
    for paper in paper_to_authors:
        print("processing paper" + str(count) + "in second for loop after" + str(time.time() - start_time) + "seconds")
        count += 1
        for author in paper_to_authors[paper]:
            for author2 in paper_to_authors[paper]:
                bar.next()
                if author != author2:
                    file.write(str(author) + "\t" + str(author2) + "\n")
    file.close()
    bar.finish()

    # file = open(filename, "w")
    # n = len(graph)
    # file.write(str(n) + "\t" + str(1) + "\n")
    # person_to_index = {}
    # for index, node in enumerate(graph):
    #     person_to_index[node] = index
    # for index, node in enumerate(graph):
    #     for neighbor in graph[node]:
    #         file.write(str(index) + "\t" + str(person_to_index[neighbor]) + "\n")
    # file.close()


# def gs_to_c_style(in_filename, out_filename):
#     # This function converts the google scholar file to a file that can be run through c code
#     start_time = time.time()
#     graph = {}
#     paper_to_authors = {}
#     with open(in_filename, 'r') as fobj:
#         lines = fobj.readlines()
#         for line in lines:
#             if line == "\n":
#                 pass
#             line = line.encode().decode('utf-8-sig')
#             data = json.loads(line)
#             print (type(data))
#             print ("processing" + data['name'] + " in first for loop after" + str(time.time() - start_time) + "seconds")
#             for paper in data['paper']:
#                 if paper['paper_id'] in paper_to_authors:
#                     paper_to_authors[paper['paper_id']].append(data['google_id'])
#                 else:
#                     paper_to_authors[paper['paper_id']] = [data['google_id']]
#
#         # Attempt to clear up some memory
#         del lines
#         del line
#         del data
#
#         for paper in paper_to_authors:
#             print ("processing in second for loop after" + str(time.time() - start_time) + "seconds")
#             for author in paper_to_authors[paper]:
#                 if author in graph:
#                     graph[author] += paper_to_authors[paper]
#                 else:
#                     graph[author] = paper_to_authors[paper]
#                 # print(graph[author])
#                 # coauthors = []
#                 # for paper in datum['paper']:
#                 #     for author in paper['author_list']:
#                 #         coauthors += ''
#                 # graph[datum['google_id']] = coauthors
#     print(graph)
#     write_graph_as_edges(graph, out_filename)
#


def writeout_clusters(graph, filename):
    with open(filename, "w") as f:
        for node_int in range(len(graph.nodes)):
            f.write(str(node_int) + ", " + str(graph.nodes[node_int]["cluster"]) + "\n")


def make_full_network_with_citations(coauthorship_filename, citations_filename):
    file = open(coauthorship_filename, "r")
    coauthor_lines = file.readlines()
    g = nx.Graph()

    name_to_citations, gender_to_citations = get_citations(citations_filename)

    # person_to_index = {}
    # index = 0
    # for index2, line in enumerate(coauthor_lines):
    #     cpy = line.split(",")
    #     if cpy[0][-1] == "\n":
    #         cpy[0] = cpy[0][:-1]
    #     if cpy[0] not in person_to_index:
    #         index += 1
    #         person_to_index[cpy[0]] = index

    counter = 0
    for line in coauthor_lines:
        counter += 1
        line = line.split(",")
        if line[-1][-1] == "\n":
            line[-1] = line[-1][:-1]
        node = line[0]
        # citations = name_to_citations[eliminate_middle_inits(line[0])]
        if line[0][2:] in name_to_citations:
            citations = name_to_citations[line[0][2:]]
        else:
            citations = -1
        g.add_node(node)
        for neighbor in line[1:]:
            neighbor_index = neighbor
            # neighbor_citations = name_to_citations[eliminate_middle_inits(neighbor)]
            if line[0][2:] in name_to_citations:
                neighbor_citations = name_to_citations[line[0][2:]]
            else:
                neighbor_citations = -1
            g.add_node(neighbor_index)
            g.add_edge(node, neighbor_index)
    return g


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
    coauthor_lines = file.readlines()  # each line is a list of coauthors for one author
    g = nx.Graph()  # undirected, no parallel edges

    # get all metadata for nodes:
    name_to_citations, gender_to_citations, phd_to_citations, name_to_phd_rank, name_to_job_rank = get_citations(
        citations_filename)

    # add all nodes  and edges to graph:
    for line in coauthor_lines:
        line = line.split(",")
        if line[-1][-1] == "\n":  # eliminates trailing newline
            line[-1] = line[-1][:-1]
        node = line[0]
        if line[0][2:] in name_to_citations:  # [2:] fixes disparity with "a/" e.g. in dblp ids
            citations = name_to_citations[line[0][2:]]
            gender = gender_to_citations[line[0][2:]]
            phd = phd_to_citations[line[0][2:]]
            phd_rank = name_to_phd_rank[line[0][2:]]
            job_rank = name_to_job_rank[line[0][2:]]
        else:
            citations = -1
            gender = 'not found'
            phd = "not found"
            phd_rank = -1
            job_rank = -1
        g.add_node(node, cluster=None, color=None, citation_count=citations, dblp_id=line[0], gender=gender, phd=phd,
                   phd_rank=phd_rank, job_rank=job_rank)
        for neighbor in line[1:]:
            neighbor_index = neighbor
            if neighbor[2:] in name_to_citations:
                neighbor_citations = name_to_citations[neighbor[2:]]
                neighbor_gender = gender_to_citations[neighbor[2:]]
                neighbor_phd = phd_to_citations[neighbor[2:]]
                neighbor_phd_rank = name_to_phd_rank[neighbor[2:]]
                neighbor_job_rank = name_to_job_rank[neighbor[2:]]
            else:
                neighbor_citations = -1
                neighbor_gender = "not found"
                neighbor_phd = "not found"
                neighbor_phd_rank = -1
                neighbor_job_rank = -1
            g.add_node(neighbor_index, cluster=None, color=None, citation_count=neighbor_citations, dblp_id=neighbor,
                       gender=neighbor_gender, phd=neighbor_phd, phd_rank=neighbor_phd_rank, job_rank=neighbor_job_rank)
            g.add_edge(node, neighbor_index)
    return g


if __name__ == "__main__":
    # gs_to_c_style('data/google_scholar.txt', 'data/google_scholar_c_style.txt')
    # graph, igraphy_graph = read_data("data/dblp/coauthorship_one_hop_out.txt")
    # igraph_ex = utils.make_graph(igraphy_graph)
    # utils.display_graph(igraph_ex, "first_20_coauthorship.png")
    # write_graph_as_edges(graph, "data//dblp/coauthorship_one_hop_c_style.txt")
    graph = make_full_network_with_citations("data/dblp/coauthorship_correct.txt", "data/dblp/dblp_id_citations")
    graph = nx.convert_node_labels_to_integers(graph)
    print(graph.number_of_nodes())
    nx.write_edgelist(graph, "data/dblp/dblp_correct.edgelist",
                      data=False)  # need to write out indices rather than ids (double check c code to confirm)
