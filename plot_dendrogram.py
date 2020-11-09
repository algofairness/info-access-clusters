#!/usr/bin/env python

# https://stackoverflow.com/questions/2455761/reordering-matrix-elements-to-reflect-column-and-row-clustering-in-naiive-python

import numpy as np
import pylab
import scipy.cluster.hierarchy as sch
import pandas as pd
import sys

def load_df(name):
    df = pd.read_csv(name, header=None)
    df = df.drop(columns=445)
    return df

def get_distance_matrix(x):
    D = np.zeros(x.shape)
    for i in range(len(x)):
        for j in range(len(x)):
            D[i,j] = sum((x[i,:] - x[j,:]) ** 2) ** 0.5
    return D

def clique_p(n, v):
    r = np.zeros((n, n))
    r[:,:] = v
    for i in range(n):
        r[i,i] = 1.0
    return r

def plot_dendro(x):
    D = get_distance_matrix(x)

    # Compute and plot dendrogram.
    fig = pylab.figure()
    axdendro = fig.add_axes([0.09,0.1,0.2,0.8])
    Y = sch.linkage(x, method='centroid')
    Z = sch.dendrogram(Y, orientation='right')
    axdendro.set_xticks([])
    axdendro.set_yticks([])

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
    index = Z['leaves']
    D = D[index,:]
    D = D[:,index]
    im = axmatrix.matshow(D, aspect='auto', origin='lower')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
    pylab.colorbar(im, cax=axcolor)

    # Display and save figure.
    fig.show()
    fig.savefig('dendrogram.png')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:\n\t%s csv-file" % sys.argv[0])
        sys.exit(1)
    plot_dendro(np.array(load_df(sys.argv[1])))
