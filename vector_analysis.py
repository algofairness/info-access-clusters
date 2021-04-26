import numpy as np
import configparser
import math
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from scipy import stats
from io import StringIO

'''
srcNodes = "input/real_input/dblp_yoj_2000_nodelist.txt"
dstVectorFile = "output_files/vectors/vectorsExp1-3_0.05_0.05.txt"
dstAnalysisFile = "output_files/analysis/analysistest.txt"
'''

def main():
    #pearson_analysis(nodelist, infile)
    #knn(srcNodes, dstVectorFile, dstAnalysisFile, 0.5, 0.5, 3, 25)
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

def zachKNN(nodefile, vecfile, analysisfile, a1, a2, neighbors, reps):
    acc_list=[]

    for i in range(reps):
        #split data
        data = split_data(nodefile, vecfile)
        Xtrain = data[0]
        ytrain = data[1]
        Xtest = data[2]
        ytest = data[3]

        #train classifier
        neigh = KNeighborsRegressor(n_neighbors=neighbors)
        neigh.fit(Xtrain, ytrain)

        #check classifier accuracy
        accuracy = test_classifier(neigh, Xtest, ytest)
        acc_list.append(accuracy)

    result = sum(acc_list)/len(acc_list)

    with open(analysisfile, 'a') as f:
        out = str(a1) + "," + str(a2) + "," #alpha1 and alpha2
        out += str(result) + ","#avg classifier accuracy
        out += vecfile + "\n" # vector files
        f.write(out)

    print("file:", vecfile, "--> accuracy:", result)
    return 1

def KNN(nodefile, vecfile, analysisfile, a1, a2, neighbors, reps):
    print("RUNNING ANALYSIS")
    cleanVecFile = clean_vectors(vecfile)
    Xvectors = np.loadtxt(cleanVecFile, delimiter=',')
    ranksLst = np.loadtxt(nodefile, delimiter='; ', skiprows=1, usecols=4)
    yranks = np.array(ranksLst)
    #make estimator/model
    neigh = KNeighborsRegressor(n_neighbors=neighbors)
    #train classifier using k-fold cross validation
    scores = cross_val_score(neigh, Xvectors, yranks, scoring="neg_root_mean_squared_error")
    result = np.average(scores)

    with open(analysisfile, 'a') as f:
        out = str(a1) + "," + str(a2) + "," #alpha1 and alpha2
        out += str(result) + ","#avg classifier accuracy
        out += vecfile + "\n" # vector files
        f.write(out)

    print("file:", vecfile, "--> average accuracy:", result)

    return 1

#returns a tuple of (Xtrain, ytrain, Xtest, ytest)
def split_data(nodefile, vecfile):
    cleanVecFile = clean_vectors(vecfile)
    Xvectors = np.loadtxt(cleanVecFile, delimiter=',')
    ranksLst = np.loadtxt(nodefile, delimiter='; ', skiprows=1, usecols=4)
    yranks = np.array(ranksLst)

    #partition process from https://stackoverflow.com/questions/3674409/how-to-split-partition-a-dataset-into-training-and-test-datasets-for-e-g-cros
    indices = np.random.permutation(Xvectors.shape[0])
    training_idx, test_idx = indices[:80], indices[80:]
    Xtrain, Xtest = Xvectors[training_idx,:], Xvectors[test_idx,:]
    ytrain, ytest = yranks[training_idx], yranks[test_idx]

    return Xtrain, ytrain, Xtest, ytest

def test_classifier(classifier, Xtest, ytest):
    acc_hits=0
    predictions = classifier.predict(Xtest)
    valueRange = np.ptp(ytest)
    errorRad = 0.1*valueRange

    for i in range(predictions.shape[0]):
        if math.dist([predictions[i]], [ytest[i]]) <= errorRad:
            acc_hits+=1

    #accuracy for k
    accuracy=acc_hits/Xtest.shape[0]

    return accuracy

#info on analysis: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
def pearson_analysis(nodefile, vecfile, analysisfile, a1, a2):
    ranksLst = np.loadtxt(nodefile, delimiter='; ', skiprows=1, usecols=4)
    ranksLst.sort()
    ranksArr = np.array(ranksLst)
    #run pca
    cleanVecFile = clean_vectors(vecfile)
    vectors = np.loadtxt(cleanVecFile, delimiter=',')
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
    #print results to file (file should be unique to experiment)
    with open(analysisfile, 'a') as f:
        out = str(a1) + "," + str(a2) + "," #alpha1 and alpha2
        out += str(result[0]) + "," + str(result[1]) + "," #correlation coef and p-value
        out += vecfile + "\n" # vector files
        f.write(out)

    print(out)
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


def clean_vector_file(filename):
    with open(filename, 'r+') as f:
        line = f.readline()
        while line:
            if line[-1] == ",":
                line.strip(",\n") + "\n"
                line = f.readline()
            else:
                line = f.readline()

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
