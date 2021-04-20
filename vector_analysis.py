import numpy as np
from sklearn.decomposition import PCA
from scipy import stats
from io import StringIO

nodelist = "input/real_input/dblp_yoj_2000_nodelist.txt"
infile = "output/vectors/vectors_experiment1-2.txt"
outfile = "output/analysis/pca.txt"
outfile1 = "output/analysis/pearson.txt"

def main():
    cleanFileObj = clean_vectors(infile)
    pearson_analysis(nodelist, cleanFileObj)
    return True


#takes as input a numpy matrix, then performs PCA analysis on it
#info on analysis: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
def pca_analysis(file):
    X = np.loadtxt(file, delimiter=',')
    print("Matrix: \n", X, "\n")
    pca2 = PCA(n_components=1)
    pca2.fit(X)
    write_pca("PCA2", pca2, outfile)
    return True

#info on analysis: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
def pearson_analysis(nodefile, vecfile):
    ranksLst = np.loadtxt(nodefile, delimiter='; ', skiprows=1, usecols=4)
    ranksArr = np.array(ranksLst)
    #run pca
    vectors = np.loadtxt(vecfile, delimiter=',')
    pca2 = PCA(n_components=1)
    pca2.fit(vectors)
    #get components from PCA
    components = np.reshape(pca2.components_, (pca2.components_.size,))

    #build a list of tuples (rank, componentVal)
    rankscomps=[]
    for i in range(ranksLst.size):
        rankscomps.append([ranksLst[i], components[i]])
    #sort rankscomps by component values
    rankscomps.sort(key=lambda t: t[1])
    #get a list of the ranks sorted by component values
    sortedRanksLst = extract_ith_tuple(rankscomps, 0)
    #make into array to run pearson
    sortedRanksArr = np.array(sortedRanksLst)
    #run the pearson analysis
    result = stats.pearsonr(ranksArr, sortedRanksArr)
    #print results to file
    with open(outfile1, 'a') as f:
        out = "PEARSON ANALYSIS ON THE FOLLOWING DATA: \n"
        out += "Node data: " + nodelist + "\n"
        out += "Vector file: " + infile + "\n"
        out += "Pearson's correlation coefficient: " + str(result[0]) + "\n"
        out += "Two-tailed p-value: " + str(result[1]) + "\n"
        out += "\n"
        f.write(out)

    return result

def extract_ith_tuple(list, i):
    out = []
    for tuple in list:
        out.append(tuple[i])
    return out

#reads in the vector file and removes trailing commas from each line
#returns a StringIO object, which behaves like a file
def clean_vectors(filename):
    out = ""
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            out += line.strip(",\n") + "\n"
            line = f.readline()
    return StringIO(out)

def write_pca(name, pca, filename):
    with open(filename, 'a') as f:
        out = "------------------------- " + name + " -------------------------\n"
        out += "n_components_:\n" + str(pca.n_components_) + "\n"
        out += "components_:\n" + str(pca.components_.shape) + "\n" + str(pca.components_) + "\n"
        out += "explained_variance_:\n" + str(pca.explained_variance_) + "\n"
        out += "explained_variance_ratio_:\n" + str(pca.explained_variance_ratio_) + "\n"
        out += "\n\n"
        f.write(out)
    return True

if __name__ == '__main__':
    main()
