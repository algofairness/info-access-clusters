from operator import mul, sub
from fractions import Fraction
from functools import reduce
import itertools
import matplotlib.pyplot as plt
import pickle
from scipy.stats import spearmanr
import os

import igraph


# Creates a picture of a graph using igraph's plotting facility
def display_graph(g, filename, layout='kk'):
    g.vs['label'] = list(range(g.vcount()))
    out = igraph.plot(g, layout=g.layout(layout))
    out.save(filename)


def make_entries(graph_dict):
    entries = []
    for i in graph_dict:
        for k in graph_dict[i]:
            entries.append({'from': i, 'to': k})

    return entries


# Input:    entries is a list of dicts, representing an edge: requires
#           {'from':id1,'to':id2}.  the ids are unique integers, not
#           necessarily consecutive
# Returns a igraph.Graph
def make_graph(entries):
    all_ids = sorted(list(set(itertools.chain.from_iterable((e['from'], e['to']) for e in entries))))
    raw_id_to_id = {raw: v for v, raw in enumerate(all_ids)}

    g = igraph.Graph(len(all_ids))

    for e in entries:
        v1, v2 = raw_id_to_id[e['from']], raw_id_to_id[e['to']]
        if not (g.are_connected(v1, v2) or v1 == v2):
            g.add_edge(v1, v2)

    h = g.induced_subgraph([i for i in range(g.vcount()) if g.degree(i) != 0])
    return h


def add_path(g, m, ind1, ind2=None):
    if m <= 0: return g
    first_new_vert = g.vcount()
    if ind2 == None:
        p = igraph.Graph(m)
        p.add_edges([(i, i + 1) for i in range(m - 1)])
        g = g + p
        g.add_edge(ind1, first_new_vert)
    elif m == 1:
        g.add_edge(ind1, ind2)
    else:
        p = igraph.Graph(m - 1)
        p.add_edges([(i, i + 1) for i in range(m - 2)])
        g = g + p
        g.add_edge(ind1, first_new_vert)
        g.add_edge(g.vcount() - 1, ind2)
    return g


# enumerates all partions of the integer n
# each output list is length of the partition, not n
def partitions(n):
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]


# enumerates all possibilities for n labeled boxes, r unlabeled balls
# length of each tuple is always n
def unlabeled_balls_labeled_boxes(n, r):
    for c in itertools.combinations_with_replacement(range(n), r):
        t = [0 for _ in range(n)]
        for i in c:
            t[i] += 1
        yield tuple(t)


# returns generator of all pairs of vertices (as indices)
# that are not edges in the input graph (not including self loops)
def non_edges(graph):
    numVerts = graph.vcount()
    if graph.is_directed():
        return ((i, j) for (i, j) in itertools.product(range(numVerts), repeat=2)
                if i != j and not graph.are_connected(i, j))
    else:
        return ((i, j) for (i, j) in itertools.combinations(range(numVerts), 2)
                if not graph.are_connected(i, j))


# defaults to strongly connected
# note vertex ids change from input graph
def get_largest_component(graph, mode='STRONG'):
    comps = graph.components(mode)
    return comps.giant()


# Does the Spearman correlation test between xs and ys
def spearman(xs, ys, return_pvalue=True):
    # make sure they're the same length and have no None's
    mlength = min(len(xs), len(ys))
    xs, ys = xs[:mlength], ys[:mlength]
    xs = [xs[i] for i in range(len(xs)) if xs[i] != None and ys[i] != None]
    ys = [ys[i] for i in range(len(ys)) if xs[i] != None and ys[i] != None]
    coeff, pval = spearmanr(xs, ys)
    if return_pvalue:
        return coeff, pval
    else:
        return coeff


# returns n choose k
def choose(n, k):
    if n < 0:
        n = 0
    if k < 0:
        k = 0
    if k == 1:
        return int(n)
    if k == 2:
        return int((n * (n - 1)) // 2)
    return int(reduce(mul, (Fraction(n - i, i + 1) for i in range(k)), 1))


def list_to_str(l):
    s = ''
    for i in l:
        s += str(i)
    return s


def memoize(f):
    cache = {}

    def memoizedFunction(*args):
        if args not in cache:
            cache[args] = f(*args)
        return cache[args]

    memoizedFunction.cache = cache
    return memoizedFunction


# Plots a time series
def plot(time_series, plot_label=None, xlabel='n', ylabel='Probability', plot_type='-', show=True):
    if plot_type == None:
        plot_type = '-'

    line, = plt.plot(range(1, len(time_series) + 1), time_series, plot_type, linewidth=1, markersize=8)

    # adds label from plot_label
    if plot_label != None:
        line.set_label(plot_label)

    x1, x2, y1, y2 = plt.axis()
    plt.axis([x1, len(time_series) + 1, y1, y2])
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if plot_label != None:
        plt.legend()
    if show:
        plt.show()
    else:
        return plt


# Plots more than one time series
def plots(time_series, plot_labels=[], xlabel='n', ylabel='probability', plot_types=[], show=True):
    if len(plot_types) == 0:
        plot_types = ['-'] * len(time_series)
    # plots lines
    lines = []
    for seq, plot_type in zip(time_series, plot_types):
        line, = plt.plot(range(1, len(seq) + 1), seq, plot_type, linewidth=1)  # , markersize=8)
        lines.append(line)

    # adds labels from plot_labels
    for line, label in zip(lines, plot_labels):
        line.set_label(label)

    x1, x2, y1, y2 = plt.axis()
    plt.axis([x1, max(len(seq) for seq in time_series) + 1, y1, y2])
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if len(plot_labels) > 0:
        plt.legend(loc='center right')
    if show:
        plt.show()
    else:
        return plt


def plots_fancy(xs, time_series, time_series_stds=None, plot_labels=[], xlabel='k', ylabel='probability', plot_types=[],
                logy=False, show=True):
    if len(plot_types) == 0:
        plot_types = ['-'] * len(time_series)

    # plots lines
    lines = []
    if time_series_stds is None:
        for seq, plot_type in zip(time_series, plot_types):
            line, = plt.plot(xs, seq, plot_type, linewidth=3)  # , markersize=8)
            lines.append(line)
    else:
        for seq, stds, plot_type in zip(time_series, time_series_stds, plot_types):
            line, = plt.plot(xs, seq, plot_type, linewidth=3)  # , markersize=8)
            plt.errorbar(xs, seq, yerr=stds, color=line.get_color(), fmt='none')  # , markersize=8)
            lines.append(line)

    if logy:
        plt.yscale('log')

    # adds labels from plot_labels
    for line, label in zip(lines, plot_labels):
        line.set_label(label)
    x1, x2, y1, y2 = plt.axis()
    # plt.axis([x1, max(len(seq) for seq in time_series)+1, y1, y2])
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if len(plot_labels) > 0:
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.02))
        # ax = plt.gca()
        # ax.legend().draggable()
    if show:
        plt.show()
    else:
        return plt


# prettyprint a matrix
def mprint(m):
    for r in m:
        print(r)


# writes obj to file given by filename
def writeObj(obj, filename):
    with open(filename, 'wb') as outfile:
        pickle.dump(obj, outfile)
    print("Pickled %s object" % filename)


# reads object and returns it from file
# given by filename
def readObj(filename):
    obj = pickle.load(open(filename, 'rb'))
    print("%s loaded." % filename)
    return obj


# shortcut to load specific data sets
def load(name):
    name_to_loc = {'prac': 'prac/prac.p',
                   'irvine': 'Data_Sets/irvine/irvine.p',
                   'email-arenas': 'Data_Sets/email_arenas/email-arenas.p',
                   'email-EU': 'Data_Sets/email_EU/email-EU.p',
                   'enron': 'Data_Sets/enron/enron_graph.p',
                   'fb': 'Data_Sets/fb/fb.p',
                   'arxiv-5': 'Data_Sets/arxiv_5/arxiv-5.p',
                   'arxiv-4': 'Data_Sets/arxiv_4/arxiv-4.p',
                   'arxiv-3': 'Data_Sets/arxiv_3/arxiv-3.p',
                   'arxiv-2': 'Data_Sets/arxiv_2/arxiv-2.p',
                   # 'hypertext':'hypertext09/hypertext09.p',
                   # 'hypertext09':'hypertext09/hypertext09.p',
                   'arxiv': 'Data_Sets/arxiv_grqc/arxiv_grqc.p',
                   'arxiv_grqc': 'Data_Sets/arxiv_grqc/arxiv_grqc.p'}
    if name not in name_to_loc:
        print("Can't find %s" % name)
    return readObj(name_to_loc[name])


# loads Ashkan's saved probabilities into a python object
# one file location for each algorithm
def load_probs(file_locs=['All_K_Probs_TIM_5', 'All_K_Probs_Greedy_5', 'All_K_Probs_Naive_5']):
    repeats = 20
    k_incr, max_k = 5, 100
    ks = [1] + [k for k in range(k_incr, max_k + 1, k_incr)]
    all_probs = [[[[] for _ in range(repeats)] for _ in ks] for _ in file_locs]
    for alg_i, file_loc in enumerate(file_locs):
        for k_i, k in enumerate(ks):
            for r in range(repeats):
                fname = '../Charts/%s/%iNumb_%iprob.txt' % (file_loc, k, r)
                if not os.path.isfile(fname):
                    fname = '../Charts/%s/%iNumb_%i_prob.txt' % (file_loc, k, r)
                with open(fname, 'r') as f:
                    probs = [float(line.rstrip('\n')) for line in f]
                    all_probs[alg_i][k_i][r] = probs
    return all_probs
