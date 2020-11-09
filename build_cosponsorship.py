import csv
import networkx as nx
import pickle
import sys
import build_generic_network as bgn
import pandas as pd
import os

import matplotlib.pyplot as plt

CONGRESS = "114"

DATA_FILE = "data_cosponsorship/govtrack_cosponsor_data_{}_congress.csv".format(CONGRESS)

FIELDNAMES = ["bills_sponsored", "bills_originally_cosponsored", "name", "thomas_id", "bioguide_id", "state",
              "district"]
FILES = ["govtrack-stats-2016-senate-cosponsored",
         "govtrack-stats-2016-senate-bills-introduced",
         "govtrack-stats-2016-senate-bills-reported-(Bills Out of Committee)",
         "govtrack-stats-2016-senate-committee-positions",
         "govtrack-stats-2016-senate-cosponsors",
         "govtrack-stats-2016-senate-transparency-bills",
         "govtrack-stats-2016-senate-ideology",
         "govtrack-stats-2016-senate-cosponsored-other-party-(Joining Bipartisan Bills)",
         "govtrack-stats-2016-senate-bills-enacted-ti",
         "govtrack-stats-2016-senate-leadership",
         "govtrack-stats-2016-senate-missed-votes",
         "govtrack-stats-2016-senate-bills-with-committee-leaders",
         "govtrack-stats-2016-senate-bills-with-companion-(Working with the Other Chamber)",
         "govtrack-stats-2016-senate-bills-with-cosponsors-both-parties-count-(Writing Bipartisan Bills)"]

MAIN_PICKLE = "output_files/strong-house_pickle"
# Attributes except for le_score
ATTRIBUTES = {"democrat": "1 = Democrat"}

def main():
    strong_cosponsorship()
    return


def weak_nested_edges(G, bill_nodes_list):
    # if len(bill_nodes_list) == 0:
    #     raise ValueError("Change of set while empty: 'ONE NODE: FALSE FALSE'")
    for u in range(len(bill_nodes_list) - 1):
        for v in range((u + 1), len(bill_nodes_list)):
            G.add_edge(bill_nodes_list[u], bill_nodes_list[v])
    return


def strong_nested_edges(G, sponsor_list, orig_cosp_list, first_push):
    if first_push:
        return
    if len(sponsor_list) < 1 and len(orig_cosp_list) > 0:
        raise ValueError(
            "bill with zero sponsors but non-zero original cosponsor(s): sponsors {}; original cosponsors {}".format(
                sponsor_list, orig_cosp_list))
    if len(sponsor_list) < 1:
        raise ValueError("zero sponsors: sponsor {}; original cosponsor(s) {}".format(sponsor_list, orig_cosp_list))
    if len(sponsor_list) > 1:
        raise ValueError(
            "more than one sponsor: sponsors {}; original cosponsor(s) {}".format(sponsor_list, orig_cosp_list))
    sponsor = sponsor_list[0]
    for i in orig_cosp_list:
        G.add_edge(sponsor, i)
    return


def strong_cosponsorship():
    G = nx.DiGraph()
    all_nodes_attributes = {}
    with open(DATA_FILE, 'r') as data_file:
        read_data_file = csv.reader(data_file)
        all_nodes = set()
        # Declaring for convention:
        current_row = next(read_data_file)
        orig_cosp_list = []
        sponsor_list = []
        fieldnames_length = len(FIELDNAMES)
        first_push = True
        for row in read_data_file:
            previous_row = current_row
            current_row = row
            if current_row[0] != previous_row[0]:
                strong_nested_edges(G, sponsor_list, orig_cosp_list, first_push)
                first_push = False
                sponsor_list = []
                orig_cosp_list = []
            if current_row[6] == "TRUE" or current_row[7] == "TRUE":
                if current_row[6] == "TRUE" and current_row[7] == "TRUE":
                    raise ValueError("legislator is both a sponsor and an original cosponsor")
                if current_row[1] not in all_nodes:
                    G.add_node(current_row[1])
                    all_nodes_attributes[current_row[1]] = {FIELDNAMES[attribute_num]: current_row[attribute_num - 1]
                                                            for attribute_num
                                                            in range(2, fieldnames_length)}
                    all_nodes_attributes[current_row[1]][FIELDNAMES[0]] = []
                    all_nodes_attributes[current_row[1]][FIELDNAMES[1]] = []
                    all_nodes.add(current_row[1])
                # Means the node is a sponsor:
                if current_row[6] == "TRUE":
                    all_nodes_attributes[current_row[1]][FIELDNAMES[0]].append([current_row[0]])
                    sponsor_list.append(current_row[1])
                # Means the node is an original cosponsor:
                if current_row[7] == "TRUE":
                    all_nodes_attributes[current_row[1]][FIELDNAMES[1]].append([current_row[0]])
                    orig_cosp_list.append(current_row[1])
        strong_nested_edges(G, sponsor_list, orig_cosp_list, first_push)
    nx.set_node_attributes(G, all_nodes_attributes)

    G = bgn.largest_connected_component_transform(G)
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    legislative_effectiveness_score(G)
    for attribute in ATTRIBUTES:
        set_attribute(G, attribute)

    with open(MAIN_PICKLE, 'wb') as pickle_file:
        pickle.dump(G, pickle_file)

    with open("output_files/strong-house_edgelist.txt", 'w') as txt_file:
        num_of_nodes = len(G.nodes)
        directed = 1
        txt_file.write("{}\t{}\n".format(num_of_nodes, directed))
        for edge in G.edges:
            txt_file.write("{}\t{}\n".format(edge[0], edge[1]))
    return

def weak_cosponsorship():
    G = nx.Graph()
    all_nodes_attributes = {}
    with open(DATA_FILE, 'r') as data_file:
        read_data_file = csv.reader(data_file)
        all_nodes = set()
        # Declaring for convention:
        current_row = next(read_data_file)
        bill_nodes_list = []
        fieldnames_length = len(FIELDNAMES)
        for row in read_data_file:
            previous_row = current_row
            current_row = row
            if current_row[0] != previous_row[0]:
                weak_nested_edges(G, bill_nodes_list)
                bill_nodes_list = []
            if current_row[6] == "TRUE" or current_row[7] == "TRUE":
                if current_row[1] not in all_nodes:
                    G.add_node(current_row[1])
                    all_nodes_attributes[current_row[1]] = {FIELDNAMES[attribute_num]: current_row[attribute_num] for attribute_num
                                                            in range(1, fieldnames_length)}
                    all_nodes_attributes[current_row[1]][FIELDNAMES[0]] = [current_row[0]]
                    all_nodes.add(current_row[1])
                else:
                    all_nodes_attributes[current_row[1]][FIELDNAMES[0]].append(current_row[0])
                bill_nodes_list.append(current_row[1])
        weak_nested_edges(G, bill_nodes_list)
    nx.set_node_attributes(G, all_nodes_attributes)

    print("Length of G:", len(G))

    G = bgn.largest_connected_component_transform(G)
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    # nx.draw_circular(G, with_labels=True)
    # plt.show()

    with open(MAIN_PICKLE, 'wb') as pickle_file:
        pickle.dump(G, pickle_file)

    with open("output_files/cosponsorship_edgelist.txt", 'w') as txt_file:
        num_of_nodes = len(G.nodes)
        directed = 0
        txt_file.write("{}\t{}\n".format(num_of_nodes, directed))
        for edge in G.edges:
            txt_file.write("{}\t{}\n".format(edge[0], edge[1]))
    return


def legislative_effectiveness_score(graph):
    print("graph length =", len(graph))
    for node in graph.nodes:
        graph.nodes[node]["edited_name"] = " ".join(graph.nodes[node]["name"].split(" ")[:2])
    bioguide_id_node_dict = {graph.nodes[node]['bioguide_id']: node for node in graph.nodes}
    print("bioguide_id_node_dict length =", len(bioguide_id_node_dict))
    edited_name_id_node_dict = {graph.nodes[node]['edited_name']: node for node in graph.nodes}
    print("name_id_node_dict length =", len(edited_name_id_node_dict))
    # test_add_att_rep(graph, bioguide_id_node_dict)

    if not os.path.isfile("CELHouse93to115.csv"):
        df = pd.read_excel(r"CELHouse93to115.xlsx")
        df.to_csv("CELHouse93to115.csv")

    with open("CELHouse93to115.csv", 'r') as data_file:
        read_data_file = csv.reader(data_file)
        name_score_dict = {}
        fieldnames_row = next(read_data_file)
        for item in range(len(fieldnames_row)):
            if fieldnames_row[item] == "Legislator name, as given in THOMAS":
                name_column = item
                print("name_column =", name_column)
            elif fieldnames_row[item] == "Congress number":
                congress_number_column = item
                print("congress_number_column =", congress_number_column)
            elif fieldnames_row[item] == "Legislative Effectiveness Score (1-5-10)":
                le_score_column = item
                print("le_score_column =", le_score_column)
        for row in read_data_file:
            if row[congress_number_column] != CONGRESS:
                continue
            # Checked: 100 members; LEScores match with the table.
            name_score_dict[" ".join(row[name_column].split(" ")[:2])] = row[le_score_column]
        # print(name_score_dict)

    if name_score_dict["Abraham, Ralph"] == 2.075010777:
        print("True")

    found_counter = 0
    not_found_for_list = []
    for member_name in edited_name_id_node_dict:
        if member_name == "Beyer, Donald":
            dataset_name = "Beyer, Don"
        elif member_name == "Pascrell, Bill,":
            dataset_name = "Pascrell, Bill"
        elif member_name == "Conyers, John,":
            dataset_name = "Conyers, John"
        elif member_name == "Knight, Stephen":
            dataset_name = "Knight, Steve"
        elif member_name == "Conaway, K.":
            dataset_name = "Conaway, Michael"
        elif member_name == "Dold, Robert":
            dataset_name = "Dold, Bob"
        elif member_name == "Hurd, Will":
            dataset_name = "Hurd, William"
        elif member_name == "Mooney, Alexander":
            dataset_name = "Mooney, Alex"
        elif member_name == "Pallone, Frank,":
            dataset_name = "Pallone, Frank"
        elif member_name == "Trott, David":
            dataset_name = "Trott, Dave"
        elif member_name == "Hill, J.":
            dataset_name = "Hill, French"
        elif member_name == "Carter, Earl":
            dataset_name = "Carter, Buddy"
        elif member_name == "Forbes, J.":
            dataset_name = "Forbes, Randy"
        else:
            dataset_name = member_name
        try:
            le_score = name_score_dict[dataset_name]
            graph.nodes[edited_name_id_node_dict[member_name]]["le_score"] = le_score
            found_counter += 1
        except:
            not_found_for_list.append((member_name, edited_name_id_node_dict[member_name]))

    print("Legislative Effectiveness score found for {}; not found for {}".format(found_counter, not_found_for_list))
    return


def set_attribute(graph, attribute):
    print("graph length =", len(graph))
    for node in graph.nodes:
        graph.nodes[node]["edited_name"] = " ".join(graph.nodes[node]["name"].split(" ")[:2])
    bioguide_id_node_dict = {graph.nodes[node]['bioguide_id']: node for node in graph.nodes}
    print("bioguide_id_node_dict length =", len(bioguide_id_node_dict))
    edited_name_id_node_dict = {graph.nodes[node]['edited_name']: node for node in graph.nodes}
    print("name_id_node_dict length =", len(edited_name_id_node_dict))
    # test_add_att_rep(graph, bioguide_id_node_dict)

    if not os.path.isfile("CELHouse93to115.csv"):
        df = pd.read_excel(r"CELHouse93to115.xlsx")
        df.to_csv("CELHouse93to115.csv")

    with open("CELHouse93to115.csv", 'r') as data_file:
        read_data_file = csv.reader(data_file)
        name_attr_dict = {}
        fieldnames_row = next(read_data_file)
        for item in range(len(fieldnames_row)):
            if fieldnames_row[item] == "Legislator name, as given in THOMAS":
                name_column = item
                print("name_column =", name_column)
            elif fieldnames_row[item] == "Congress number":
                congress_number_column = item
                print("congress_number_column =", congress_number_column)
            elif fieldnames_row[item] == ATTRIBUTES[attribute]:
                attribute_column = item
                print("{}_column =".format(attribute), attribute_column)
        for row in read_data_file:
            if row[congress_number_column] != CONGRESS:
                continue
            # Checked: 100 members; LEScores match with the table.
            name_attr_dict[" ".join(row[name_column].split(" ")[:2])] = row[attribute_column]
        # print(name_score_dict)

    found_counter = 0
    not_found_for_list = []
    for member_name in edited_name_id_node_dict:
        if member_name == "Beyer, Donald":
            dataset_name = "Beyer, Don"
        elif member_name == "Pascrell, Bill,":
            dataset_name = "Pascrell, Bill"
        elif member_name == "Conyers, John,":
            dataset_name = "Conyers, John"
        elif member_name == "Knight, Stephen":
            dataset_name = "Knight, Steve"
        elif member_name == "Conaway, K.":
            dataset_name = "Conaway, Michael"
        elif member_name == "Dold, Robert":
            dataset_name = "Dold, Bob"
        elif member_name == "Hurd, Will":
            dataset_name = "Hurd, William"
        elif member_name == "Mooney, Alexander":
            dataset_name = "Mooney, Alex"
        elif member_name == "Pallone, Frank,":
            dataset_name = "Pallone, Frank"
        elif member_name == "Trott, David":
            dataset_name = "Trott, Dave"
        elif member_name == "Hill, J.":
            dataset_name = "Hill, French"
        elif member_name == "Carter, Earl":
            dataset_name = "Carter, Buddy"
        elif member_name == "Forbes, J.":
            dataset_name = "Forbes, Randy"
        else:
            dataset_name = member_name
        try:
            attribute_value = name_attr_dict[dataset_name]
            graph.nodes[edited_name_id_node_dict[member_name]][attribute] = attribute_value
            found_counter += 1
        except:
            not_found_for_list.append((member_name, edited_name_id_node_dict[member_name]))

    print("{} found for {}; not found for {}".format(attribute, found_counter, not_found_for_list))
    return


def test_add_att_sen(graph, bioguide_id_node_dict):
    test_cases = [("T000476", "Tillis"),
                 ("M000303", "McCain"),
                 ("I000024", "Inhofe"),
                 ("M001111", "Murray"),
                 ("B000575", "Blunt")]
    counter = 0
    for test_case in test_cases:
        counter += 1
        node_num = bioguide_id_node_dict[test_case[0]]
        if graph.nodes[node_num]["name"].split(", ")[0] == test_case[1]:
            print("Test Case {}: Successful".format(counter))
        else:
            print("Test Case {}: Successful".format(counter))


def csv_nodes(G):
    fieldnames = ["name", "bioguide_id", "state", "district", "democrat", "le_score"]
    with open("output_files/cosponsorship_graph_nodes.csv", 'a') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        user_obj_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        counter = 0
        not_found = []
        for node in G.nodes:
            try:
                row = [G.nodes[node][i] for i in fieldnames]
                user_obj_writer.writerow(row)
                counter += 1
            except:
                not_found.append(G.nodes[node][fieldnames[0]])
        print("counter =", counter)
        print("not_found =", not_found)
    return


def analyze_the_graph(G):
    temp_graph = nx.Graph()
    for edge in G.edges:
        temp_graph.add_edge(edge[0], edge[1])
    smallest_cc = min(nx.connected_components(temp_graph), key=len)

    other_graph = G.subgraph(smallest_cc).copy()
    cc = nx.strongly_connected_components(other_graph)

    print([len(i) for i in cc])
    return

if __name__ == "__main__":
    main()
