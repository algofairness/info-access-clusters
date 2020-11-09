#!/usr/bin/env python

import sklearn.metrics as sm
import pandas as pd
import numpy as np
import sys
import pylab

def compute_clustering_similarity_map(clusterings):
    n = clusterings.shape[1]
    result = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            result[i, j] = sm.adjusted_rand_score(clusterings[:,i], clusterings[:,j])
    return result

if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1])
    df = df.iloc[:,1:]
    d = compute_clustering_similarity_map(np.array(df))
    pylab.matshow(d)
    pylab.colorbar()
    pylab.show()
