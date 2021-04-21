#implementation from https://www.python-graph-gallery.com/heatmap/ and
#https://www.kite.com/python/docs/seaborn.heatmap and
#https://blog.quantinsti.com/creating-heatmap-using-python-seaborn/
import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#datafile = "output_files/analysis/analysis1-3.txt"
#outfile = "output_files/analysis/testimage.png"

def main():
    #heatmap(datafile, outfile)
    return 1

def heatmap(datafile, outfile):

    #data = np.loadtxt(datafile, delimiter=',', usecols = (0,1,2))
    df = pd.read_csv(datafile, header=1, usecols=[0,1,2,3])
    result = df.pivot(index='alpha1', columns='alpha2', values='correlation')
    print(result)

    seaborn.heatmap(result)
    plt.savefig(outfile)

main()
