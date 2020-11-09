import matplotlib.pyplot as plt
import numpy as np
import seaborn
import csv
import sys
import statistics
import numpy
import pylab as pl
from matplotlib import collections  as mc
import random

# The noise variable controls the displacement of new lines from the cluster center line so that
# large clusters are visibly larger.  This number is set in an ad hoc way based on the dataset
# size so that clusters aren't so large they run into each other and aren't so small the size
# isn't visible.
NOISE = 0.0001

TICKSIZE = 15
FONTSIZE = 20
TITLESIZE = 30
plt.rcdefaults()

def plot_line_segments(line_segments_lol):
    lc = mc.LineCollection(line_segments_lol, alpha=0.01)
    fig, ax = pl.subplots()
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    plt.show()

def get_clustid_position(clustid, count):
    # return clustid + (count - 1) * NOISE * ((-1) ** count)
    #return clustid
    return random.normalvariate(clustid, 0.1)

def get_alpha_position(alpha, count):
    return random.normalvariate(alpha, 0.005)
    # return alpha + (count - 1) * NOISE * ((-1) ** count)
    # return alpha

def read_cluster_alpha_file(filename):
    """
    Given a CSV file where rows are points, columns are alpha values, and entries are cluster
    labels.
    Returns two dictionaries of dictionaries:
    - alpha -> cluster label -> list of ids in the cluster
    - pointid -> alpha -> cluster label

    CSV input example:
    id,0.1,0.2,0.3
    1,0,0,0
    2,0,0,0
    3,0,0,0
    4,1,1,1
    5,1,1,1
    """
    all_lines = []
    alpha_clustlabel_dict = {}
    pointid_alpha_clust_dict = {}
    with open (filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pointid = ''
            line_segments = []
            for alpha, clust in row.items():
                if pointid == '':
                    pointid = clust
                    continue
                alpha = float(alpha)
                clust = int(clust)

                if not alpha in alpha_clustlabel_dict:
                    alpha_clustlabel_dict[alpha] = {}
                if not clust in alpha_clustlabel_dict[alpha]:
                    alpha_clustlabel_dict[alpha][clust] = []
                alpha_clustlabel_dict[alpha][clust].append(pointid)

                if not pointid in pointid_alpha_clust_dict:
                    pointid_alpha_clust_dict[pointid] = {}
                pointid_alpha_clust_dict[pointid][alpha] = clust
        return alpha_clustlabel_dict, pointid_alpha_clust_dict

def read_externalinfo_file(filename, headerid):
    with open (filename) as csvfile:
        externalinfo = {}
        reader = csv.DictReader(csvfile)
        for row in reader:
            idval = ''
            for header, value in row.items():
                if header == headerid:
                    idval = value
                    externalinfo[idval] = {}
                else:
                    externalinfo[idval][header] = value
        return externalinfo

def graph_alpha_v_clustid(pointid_alpha_clust_dict):
    """
    Input: a dictionary mapping from pointid -> alpha -> cluster id
    This function displays a graph where the x-axis is alpha, the y-axis is the cluster
    label, and each line is a point.
    """
    alpha_clust_count = {}
    all_lines = []
    for pointid in pointid_alpha_clust_dict:
        line_segments = []
        for alpha in pointid_alpha_clust_dict[pointid]:
            clustlabel = pointid_alpha_clust_dict[pointid][alpha]

            if not alpha in alpha_clust_count:
                alpha_clust_count[alpha] = {}
            if not clustlabel in alpha_clust_count[alpha]:
                alpha_clust_count[alpha][clustlabel] = 0
            alpha_clust_count[alpha][clustlabel] += 1

            # x = get_alpha_position(alpha, alpha_clust_count[alpha][clustlabel])
            # y = get_clustid_position(clustlabel, alpha_clust_count[alpha][clustlabel])
            x, y = numpy.random.multivariate_normal([alpha, clustlabel],
                                                    [[0.001, 0],[0,0.005]])
            line_segments.append([x, y])
        all_lines.append(line_segments)
    plot_line_segments(all_lines)

def string_to_list(stringlist):
    stringlist = stringlist[1:-1]
    parts = stringlist.split(",")
    num_parts = [ int(x) for x in parts ]
    return num_parts

def graph_alpha_v_externalinfo(alpha_clustlabel_dict, pointid_alpha_clust_dict, externalinfo):
    all_lines = []
    for pointid in pointid_alpha_clust_dict:
        line_segments = []
        for alpha in pointid_alpha_clust_dict[pointid]:
            clustlabel = pointid_alpha_clust_dict[pointid][alpha]
            cluster = alpha_clustlabel_dict[alpha][clustlabel]
            cluster = [ int(x) for x in cluster ]

            x = random.normalvariate(alpha, 0.005)
            citationcount_avg = statistics.mean(cluster)
            y = random.normalvariate(citationcount_avg, 50)

            line_segments.append([x, y])
        all_lines.append(line_segments)
    plot_line_segments(all_lines)

## TODO
# then make one graphing function that does graphs with y-axis meaning the cluster position
# and another that reads in additional data mapping from id to actual information and groups
# cluster by real data position.


def save_plot(xtitle, ytitle, figtitle, filename, plot):
    """
    This function isn't currently used, but a modified version of it could be useful for saving
    the generated image to file.
    """
    xlabel = plt.xlabel(xtitle, fontsize = FONTSIZE)
    ylabel = plt.ylabel(ytitle, fontsize = FONTSIZE)
    title = plt.title(figtitle, fontsize = TITLESIZE)
    plt.xticks(fontsize = TICKSIZE)
    plt.yticks(fontsize = TICKSIZE)

    box = plot.get_position()
    plot.set_position([box.x0, box.y0, box.width * 0.85, box.height]) # resize position

    # Put a legend to the right side
    legend = plot.legend(loc='lower center', bbox_to_anchor=(1.7, 0.5), ncol=1,
                         prop={'size': FONTSIZE})

    bb = (ylabel, ylabel, title, legend)

    plot.figure.savefig(filename, bbox_extra_artists = bb, bbox_inches = 'tight')
    print("plot saved to:" + filename)
    plt.clf()
    plt.close()

# MAIN: assumes the labeling file is given as input on the command line, e.g.:
# python3 cluster_graphing.py test_labels.csv
alpha_clustlabel_dict, pointid_alpha_clust_dict = read_cluster_alpha_file(sys.argv[1])
# externalinfo = read_externalinfo_file(sys.argv[2], 'network_id')
# graph_alpha_v_externalinfo(alpha_clustlabel_dict, pointid_alpha_clust_dict, externalinfo)
graph_alpha_v_clustid(pointid_alpha_clust_dict)
