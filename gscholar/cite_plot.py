import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import stats

import matplotlib.cm as cm

from decimal import Decimal

# Parameters for finding K:
MIN_CLUSTERS = 1
MAX_CLUSTERS = 10
N_REFS = 4

# (Optional) Colors used for the graphs.
COLOR_PALETTE = ["#FFC107", "#1E88E5", "#2ECE54", "#EC09D7", "#DDEC4E", "#D81B50", "#CCD85D", "#3701FA", "#D39CA7", "#27EA9F", "#5D5613", "#DC6464"]

def main():
    citefile = sys.argv[1]
    clusterfile = sys.argv[2]
    K = int(sys.argv[3])
    
    citations = []
    file = open(citefile, "r")
    for line in file:
        s = int(line.split(None, 2)[0])
        t = int(line.split(None, 2)[1])
        citations.append(t)
    file.close()
    
    n = 0
    clusters = []
    file = open(clusterfile, "r")
    for line in file:
        s = int(line.split(None, 2)[0])
        t = int(line.split(None, 2)[1])
        clusters.append(t)
        n += 1
    file.close()
    
    clusters_total = {cluster: [] for cluster in range(K)}
    for i in range(0, n):
        cluster = clusters[i]
        value = float(citations[i])
        clusters_total[cluster].append(value)
        
    print((len(clusters_total[0]) + len(clusters_total[1])) / float(n))
    #print(clusters_total[0])
    #print(clusters_total[1])
    #test_output = stats.kruskal(clusters_total[0][-93506:], clusters_total[1][-92134:])
    #stat, pval = stats.kruskal(clusters_total[0], clusters_total[1])
    #print('p-value =','{:.20e}'.format(Decimal(pval)))
    #print(pval)
    #print(str(test_output) + "\n")
            
    plt.figure(figsize=(12, 10))
    color_counter = 0
    #for cluster in clusters_total:
    
    
    for cluster in clusters_total:
        sns.distplot(clusters_total[cluster], hist = False, kde = True, hist_kws = {'linewidth': 3}, label = str(cluster), norm_hist = True, color=COLOR_PALETTE[color_counter])

        color_counter += 1
        
    # Runs and writes the results of Pairwise Kolmogorov-Smirnov and Kruskal-Wallis tests."""
    #kolmogorov_smirnov_test(clusters_total, K)
    kruskal_wallis_test(clusters_total, K)
        
    plt.xlabel("citations")
    plt.ylabel("PDF")
    plt.title("Cluster Citation Density")
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.savefig("Cluster_Citation_Density" + str(K), bbox_inches='tight')
    
def kruskal_wallis_test(clusters_total, K):
    """Runs and writes the results of Kruskal-Wallis test."""
    arg_list = [clusters_total[i] for i in range(K)]
    print("\nkruskal-wallis, {}-clusters:\n".format(K))
    
    #print(*arg_list)
    #print(arg_list)

    test_output = stats.kruskal(*arg_list)
    stat, pval = stats.kruskal(*arg_list)
    print('p-value =','{:.20e}'.format(Decimal(pval)))
    print(str(test_output) + "\n")
    return
    
def kolmogorov_smirnov_test(clusters_total, K):
    """Runs and writes the results of Pairwise Kolmogorov-Smirnov test."""
    for i in range(K):
        current_num = K - 1 - i
        for j in range(current_num):
            print("\n{} to {}".format(j, current_num))

            test_output = stats.ks_2samp(clusters_total[j], clusters_total[current_num])
            print("\n" + str(test_output))
    return

if __name__ == "__main__":
    main()
