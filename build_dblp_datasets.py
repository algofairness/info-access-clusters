import networkx as nx
import pandas as pd
import csv
import build_generic_network as bgn
import os

YEARS_OF_JOB = [2000, 2001, 2002, 2003, 2004, 2005, 2021]
NODES_PATH = "dblp_data/faculty_data.xlsx"
PUBLICATIONS_PATH = "dblp_data/processed_publications.csv"
NODE_ATTRIBUTES_TO_PROCESS = ['name', 'gender', 'year_of_job', 'dblp_id', "gs_id", "followup_title",
                              "followup_location", "followup_department"]
NODE_ATTRIBUTES_TO_PRODUCE = ['node', 'dblp_id', 'gender', 'phd', 'phd_rank', 'job_rank']
NON_UNIQUE_AUTH_PATH = "dblp_data/non_unique_auth.txt"

def read_nodelist():
    # Uses NODES_PATH, NODE_ATTRIBUTES_TO_PROCESS
    node_dict = {}

    df_faculty = pd.read_excel(NODES_PATH, "faculty")
    for index, row in df_faculty.iterrows():
        node = preprocess_id(row['dblp_id'])
        if node == -1:
            continue
        node_dict[node] = {}
        node_dict[node]["excel_index"] = index
        for attribute in NODE_ATTRIBUTES_TO_PROCESS:
            node_dict[node][attribute] = row[attribute]
        node_dict[node]["phd"] = row["location_phd"]
        node_dict[node]["job"] = row["location_job"]
        node_dict[node]["phd_rank"] = university_to_rank(node_dict[node]["phd"])
        node_dict[node]["job_rank"] = university_to_rank(node_dict[node]["job"])

    # with open(NODES_PATH, 'r') as file:
    #     first = 1
    #     for line in file:
    #         if first:
    #             first -= 1
    #             continue
    #         if line[-1] == "\n":
    #             line = line[:-1]
    #         line = line.split("; ")
    #
    #         node = pre_process_dblp_id(line[2])
    #         node_dict[node] = {}
    #
    #         for i in range(len(NODE_ATTRIBUTES_TO_PROCESS)):
    #             node_dict[node][NODE_ATTRIBUTES_TO_PROCESS[i]] = line[i]
    print("len(node_dict) =", len(node_dict))
    return node_dict


def preprocess_id(name):
    try:
        output = name.split(":")
    except:
        return -1

    if output == ["#NAME?"] or output == ["#ERROR!"]:
        return -1

    if len(output) > 2:
        raise ValueError("len(output) > 2")

    if len(output) == 1:
        raise ValueError("len(output) == 1")

    output[0], output[1] = output[1], output[0]
    output = "".join(output)

    function_output = []
    for i in output:
        if i.isalpha() or i.isnumeric():
            function_output.append(i)
    function_output = "".join(function_output)
    return function_output


def university_to_rank(university):
    '''
    Takes in the name of a university and returns its rank
    (according to the ranking system described in https://advances.sciencemag.org/content/1/1/e1400005)
    '''
    with open("dblp_data/faculty_data - schools.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0] == university:
                return float(row[2])
        return -1


def populate_with_nodes(graph, node_dict):
    for node in node_dict:
        graph.add_node(node)
    nx.set_node_attributes(graph, node_dict)

    find_non_unique_auth(graph)

    print("Before removal of non-unique auth: len(graph.nodes)=", len(graph.nodes))
    remove_non_unique_auth(graph)
    print("After removal of non-unique auth: len(graph.nodes)=", len(graph.nodes))
    return graph


def find_non_unique_auth(graph):
    """
    Finds those authors in the xml file whose raw names are mapped to the same string after preprocessing.
    :param graph: nx.Graph
    :return: None
    """
    if not os.path.isfile(NON_UNIQUE_AUTH_PATH):
        with open(NON_UNIQUE_AUTH_PATH, 'w') as file_to_write:
            all_proc_authors = set()
            all_raw_authors = set()
            non_unique_authors = set()
            with open(PUBLICATIONS_PATH, 'r') as file_to_read:
                csv_reader = csv.reader(file_to_read, delimiter=',')
                starting = 1
                for row in csv_reader:
                    if starting:
                        starting -= 1
                        continue
                    publication_type, year, num_of_auth, author, title = parse_publication(row)
                    if num_of_auth > 1:
                        for a in author:
                            p_name = preprocess_name(a)
                            # Checking of pigeonholing:
                            if p_name in all_proc_authors and a not in all_raw_authors:
                                non_unique_authors.add(p_name)
                            all_proc_authors.add(p_name)
                            all_raw_authors.add(a)
            print("len(all_proc_authors) =", len(all_proc_authors))
            unique_authors = all_proc_authors - non_unique_authors
            authors_of_interest = set(graph.nodes.keys())
            print("Check: {} and {}".format(len(graph), len(authors_of_interest)))
            # Pruning our authors of interest from the faculty data
            # by removing the non-unique authors found in their set:
            for a in authors_of_interest:
                # Finds non-unique authors of interest:
                if a not in unique_authors:
                    file_to_write.write(a + "\n")
    return


def remove_non_unique_auth(graph):
    with open(NON_UNIQUE_AUTH_PATH, 'r') as file:
        for node in file:
            if node[-1] == "\n":
                node = node[:-1]
            graph.remove_node(node)
    return


def populate_with_edges(graph, year_of_job):
    # Uses PUBLICATIONS_PATH
    # inclusive of yoj-1
    with open(PUBLICATIONS_PATH, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        start = 1
        for row in csv_reader:
            if start:
                start -= 1
                continue

            publication_type, year, num_of_auth, author, title = parse_publication(row)

            # Exclude the publications with no year specified or two years specified
            # (as was in one single-author publication, manually validated against dblp as having two years):
            if year is None:
                continue
            if year < year_of_job and num_of_auth > 1:
                for i in range(len(author) - 1):
                    for j in range(i + 1, len(author)):
                        author_i = preprocess_name(author[i])
                        author_j = preprocess_name(author[j])
                        if graph.has_node(author_i) and graph.has_node(author_j):
                            graph.add_edge(author_i, author_j)
    print("populate_with_edges len(graph) =", len(graph))
    return graph


def preprocess_name(name):
    function_output = []
    for i in name:
        if i.isalpha() or i.isnumeric():
            function_output.append(i)
    function_output = "".join(function_output)
    # name_segments = name.split(" ")
    # reordered_name_segments = [name_segments[-1]]
    # reordered_name_segments.extend(name_segments[:-1])
    #
    # print(reordered_name_segments)
    # name_list = []
    # for segment in reordered_name_segments:
    #     sub_string = []
    #     for i in segment:
    #         if i.isalpha():
    #             sub_string.append(i)
    #         else:
    #             sub_string.append("=")
    #     name_list.append("".join(sub_string))
    #
    # finalized_substrings = []
    # for i in range(len(name_list)):
    #     if i < 2:
    #         if i == 0:
    #             finalized_substrings.append(name_list[i] + ":")
    #         else:
    #             finalized_substrings.append(name_list[i])
    #     else:
    #         if i == 2:
    #             finalized_substrings.append("_" + name_list[i])
    #         else:
    #             finalized_substrings.append(name_list[i])
    # print(finalized_substrings)
    # print("".join(finalized_substrings))
    return function_output


def parse_publication(row):
    publication_type = row[0]
    try:
        year = int(row[1][1:-1])
    except:
        year = None
    try:
        num_of_auth = int(row[2])
    except:
        raise ValueError("error")
    author = row[3][1:-1].replace("'", "").split(", ")
    title = row[4][2:-2]
    return publication_type, year, num_of_auth, author, title


def make_edgelist(graph, output_name):
    if graph.is_multigraph():
        raise TypeError("Graph has parallel edges")

    if graph.is_directed():
        directed = 1
    else:
        directed = 0

    with open(output_name, 'w') as txt_file:
        num_of_nodes = len(graph.nodes)
        txt_file.write("{}\t{}\n".format(num_of_nodes, directed))
        for edge in graph.edges:
            txt_file.write("{}\t{}\n".format(edge[0], edge[1]))
    return


def make_nodelist(graph, output_name, attribute_list):
    # Requires attribute_list to have "node" at index 0.
    with open(output_name, 'w') as txt_file:
        header_string = "; ".join(attribute_list)
        txt_file.write(header_string + "\n")

        for node in range(len(graph.nodes)):
            row = [node]
            for a in attribute_list[1:]:
                row.append(graph.nodes[node][a])
            str_row = [str(i) for i in row]
            line_string = "; ".join(str_row)
            txt_file.write(line_string + "\n")
    return


def convert_graphs_to_json():
    for year in YEARS_OF_JOB:
        graph = nx.Graph()
        edgelist_name = "../information-access-clustering/dblp_data/datasets_by_yoj/dblp_yoj_{}_edgelist.txt".format(year)
        nodelist_name = "../information-access-clustering/dblp_data/datasets_by_yoj/dblp_yoj_{}_nodelist.txt".format(year)

        # Populate graph with edges:
        with open(edgelist_name, 'r') as edges_file:
            first_line = True
            for line in edges_file:
                if first_line:
                    first_line = False
                    continue
                if line[-1] == "\n":
                    line = line[:-1]
                line = [int(i) for i in line.split("\t")]
                graph.add_edge(line[0], line[1])

        # Assign the attributes to the nodes:
        node_to_attr = {}
        with open(nodelist_name, 'r') as nodes_file:
            first_line = True
            for line in nodes_file:
                if first_line:
                    first_line = False
                    fields = line[:-1].split("; ")
                    continue
                if line[-1] == "\n":
                    line = line[:-1]
                line = line.split("; ")

                row = []
                for i in range(len(line)):
                    if i == 0:
                        row.append(int(line[i]))
                    else:
                        try:
                            row.append(float(line[i]))
                        except:
                            row.append(line[i])

                node = row[0]
                node_to_attr[node] = {}
                for i in range(1, len(fields)):
                    node_to_attr[node][fields[i]] = row[i]
        nx.set_node_attributes(graph, node_to_attr)
        print("Graph: {} nodes, {} edges".format(len(graph.nodes), len(graph.edges)))

        # Jsonify the networkx object:
        output_path = "../information-access-clustering/dblp_data/datasets_by_yoj/dblp_yoj_{}.json".format(year)
        bgn.graph_to_json(graph, output_path)
    return


def main():
    if os.path.isfile(NON_UNIQUE_AUTH_PATH):
        raise ValueError("File already exists at NON_UNIQUE_AUTH_PATH")

    for year_of_job in YEARS_OF_JOB:
        graph = nx.Graph()
        edgelist_name = "dblp_data/datasets_by_yoj/dblp_yoj_{}_edgelist.txt".format(year_of_job)
        nodelist_name = "dblp_data/datasets_by_yoj/dblp_yoj_{}_nodelist.txt".format(year_of_job)

        node_dict = read_nodelist()
        graph = populate_with_nodes(graph, node_dict)
        graph = populate_with_edges(graph, year_of_job)
        graph = bgn.largest_connected_component_transform(graph)
        graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
        make_edgelist(graph, edgelist_name)

        attribute_list = NODE_ATTRIBUTES_TO_PRODUCE
        keys = {key for key in graph.nodes[0]}
        for a in attribute_list:
            try:
                keys.remove(a)
            except:
                continue
        for key in keys:
            attribute_list.append(key)
        make_nodelist(graph, nodelist_name, attribute_list)
        print("Dataset created for year_of_job = {} with {} nodes and {} edges\n".format(year_of_job, len(graph.nodes), len(graph.edges)))
    convert_graphs_to_json()
    return


if __name__ == '__main__':
    main()
