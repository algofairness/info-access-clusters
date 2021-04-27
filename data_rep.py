#implementation from https://www.python-graph-gallery.com/heatmap/ and
#https://www.kite.com/python/docs/seaborn.heatmap and
#https://blog.quantinsti.com/creating-heatmap-using-python-seaborn/
import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#datafile = "output_files/analysis/analysis1-5.txt"
#outfile = "output_files/analysis/heatmap1-5.png"

def main():
    #heatmap(datafile, outfile)
    return 1

def pcaHeatmap(datafile, outfile):
    df = pd.read_csv(datafile, header=1, usecols=[0,1,2,3])
    shapedDF = df.pivot(index='alpha2', columns='alpha1', values='correlation')
    print(shapedDF)
    heatmap=seaborn.heatmap(shapedDF)
    heatmap.invert_yaxis()
    plt.savefig(outfile)
    return

def zachKNNHeatmap(datafile, outfile):
    df = pd.read_csv(datafile, header=1, usecols=[0,1,2])
    shapedDF = df.pivot(index='alpha2', columns='alpha1', values='accuracy')
    print(shapedDF)
    heatmap=seaborn.heatmap(shapedDF)
    heatmap.invert_yaxis()
    plt.savefig(outfile)
    return

def KNNHeatmap(datafile, outfile):
    df = pd.read_csv(datafile, header=1, usecols=[0,1,2])
    shapedDF = df.pivot(index='alpha2', columns='alpha1', values='accuracy')
    print(shapedDF)
    heatmap=seaborn.heatmap(shapedDF)
    heatmap.invert_yaxis()
    plt.savefig(outfile)
    return

def randomForestHeatmap(datafile, outfile):
    df = pd.read_csv(datafile, header=1, usecols=[0,1,2,3])
    shapedDF = df.pivot(index='alpha2', columns='alpha1', values='mean')
    print(shapedDF)
    heatmap=seaborn.heatmap(shapedDF)
    heatmap.invert_yaxis()
    plt.savefig(outfile)
    return

def SVRHeatmap(datafile, outfile):
    df = pd.read_csv(datafile, header=1, usecols=[0,1,2,3])
    shapedDF = df.pivot(index='alpha2', columns='alpha1', values='mean')
    print(shapedDF)
    heatmap=seaborn.heatmap(shapedDF)
    heatmap.invert_yaxis()
    plt.savefig(outfile)
    return

main()
