import csv
import pandas as pd
import random
import sys

from clustering import Clustering

def string_to_list(stringlist):
    stringlist = stringlist[1:-1]
    print(stringlist)
    parts = stringlist.split(",")
    print(parts)
    num_parts = [ int(x) for x in parts ]
    return num_parts

def read_clustering_file(filename):
    """
    Given a CSV file where rows are alpha values and entries are cluster lists this returns
    a list of alpha values and a list of lists of clustering objects.

    CSV input example:
    alpha,c1,c2
    0.1,"[1,2,3]","[4,5]"
    0.2,"[4,5]","[1,2,3]"
    0.3,"[1,2,3]","[4,5]"
    """
    alpha_clustering_lol = []
    with open (filename) as csvfile:
        reader = csv.DictReader(csvfile)
        cluster_names = []
        alphas = []
        for row in reader:
            alpha = ''
            alpha_clusters = []
            for clustname, clustlist in row.items():
                if alpha == '':
                    alpha = clustlist
                else:
                    clustlist = string_to_list(clustlist)
                    alpha_clusters.append(clustlist)
            clustering = Clustering(alpha, alpha_clusters)
            alphas.append(alpha)
            alpha_clustering_lol.append(clustering)
        return alphas, alpha_clustering_lol

def clustering_to_labeling(alpha_clustering_lol):
    last = None
    for clustering in alpha_clustering_lol:
        if last == None:
            last = clustering
        else:
            print("---------")
            Clustering.set_labeling_maxmatching(last, clustering)
            last = clustering

def write_clustering_labels_to_file(alphas, alpha_clustering_lol, filename):
    id_alpha_label_dod = {}
    for alpha, clustering in zip(alphas, alpha_clustering_lol):
        dict_id_labels = clustering.get_dict_id_labels()
        for pointid in dict_id_labels:
            if not pointid in id_alpha_label_dod:
                id_alpha_label_dod[pointid] = {'id': pointid}
            id_alpha_label_dod[pointid][alpha] = dict_id_labels[pointid]

    f = open(filename, "w")
    with open(filename, 'w') as csvfile:
        colnames = ['id'] + alphas
        writer = csv.DictWriter(csvfile, fieldnames = colnames)
        writer.writeheader()
        for pointid in id_alpha_label_dod:
            writer.writerow(id_alpha_label_dod[pointid])


def main(clustering_file, labeling_file):
    # MAIN: takes as input a clustering file and outputs a labeling file
    print(clustering_file)
    alphas, alpha_clustering_lol = read_clustering_file(clustering_file)
    clustering_to_labeling(alpha_clustering_lol)
    write_clustering_labels_to_file(alphas, alpha_clustering_lol, labeling_file)
    return
