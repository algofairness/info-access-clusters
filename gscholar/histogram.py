import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import stats

import matplotlib.cm as cm

def main():
    probfile = sys.argv[1]
    
    file = open(probfile, "r")
    nodes = file.readlines()
    all_data = []
    for index, line in enumerate(nodes):
        line = line.split(",")
        #print("the length of the line is ", len(line))
        for prob in line:
            all_data.append(float(prob))
    file.close()

    plt.hist(all_data)
    plt.xlabel("value at p_ij")
    plt.ylabel("count")
    plt.yscale('log', nonposy='clip')
    plt.title("Composition of probabilities in information access vectors")
    #plt.autoscale(enable=True, axis='both', tight=None)
    plt.savefig("probability_composition.png", bbox_inches='tight')
    plt.close()
        
    return

if __name__ == "__main__":
    main()
