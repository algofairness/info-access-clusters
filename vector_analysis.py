import numpy as np
from numpy import mean
from numpy import std
import time
import configparser
import math
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
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
    print(a1,a2)
    print("file:", vecfile, "--> accuracy:", result)
    return 1

def KNN(nodefile, vecfile, analysisfile, a1, a2, neighbors, reps):
    print("Running KNN analysis...")
    cleanVecFile = clean_vectors(vecfile)
    Xvectors = np.loadtxt(cleanVecFile, delimiter=',')
    ranksLst = np.loadtxt(nodefile, delimiter='; ', skiprows=1, usecols=4)
    yranks = np.array(ranksLst)
    #make estimator/model
    neigh = KNeighborsRegressor(n_neighbors=neighbors)
    #train classifier using k-fold cross validation
    scores = cross_val_score(neigh, Xvectors, yranks, scoring="neg_root_mean_squared_error")

    with open(analysisfile, 'a') as f:
        out = str(a1) + "," + str(a2) + "," #alpha1 and alpha2
        out += str(mean(scores)) + "," + str(std(scores)) + ","#avg classifier accuracy
        out += vecfile + "\n" # vector files
        f.write(out)

    print("[a1, a2] = [", a1, ", ", a2, "]: average accuracy=", mean(scores), " std=", std(scores))

    return mean(scores), std(scores)

def holdoutKNN(nodefile, vecfile, holdnodefile, holdvecfile, analysisfile, a1, a2, neighbors, components):
    print("Running KNN Holdout Analysis...")
    X_train, y_train = make_data(nodefile, vecfile)
    X_test, y_test = make_data(holdnodefile, holdvecfile)
    # intialize pca and knn,dummy regression models
    # from https://towardsdatascience.com/principal-component-analysis-for-dimensionality-reduction-115a3d157bad
    pca = PCA(n_components=components)
    knn = KNeighborsRegressor(n_neighbors=neighbors)
    # fit and transform data
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.fit_transform(X_test)
    #fit to models
    knn.fit(X_train_pca, y_train)
    #make predictions on test data
    knn_pred = knn.predict(X_test_pca)
    #get scores
    knn_score = mean_squared_error(y_test, knn_pred, squared=False)
    print("knn score: ", knn_score)

    return knn_score


def randomForest(nodefile, vecfile, analysisfile, a1, a2):
    # evaluate random forest ensemble for regression
    # define dataset
    start = time.time() #beginning time
    X, y = make_data(nodefile, vecfile)
    # define the model
    model = RandomForestRegressor()
    # evaluate the model
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    #scoring was originally 'neg_mean_absolute_error'
    n_scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    end = time.time() #end time

    with open(analysisfile, 'a') as f:
        out = str(a1) + "," + str(a2) + ","
        out += str(mean(n_scores)) + "," + str(std(n_scores)) + ","
        out += vecfile + "\n"
        f.write(out)
    print('a1, a2 =', a1, ',', a2, ' MSE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)), "time: ", end-start)

    return mean(n_scores), std(n_scores)

def holdoutRandomForest(nodefile, vecfile, holdnodefile, holdvecfile, analysisfile, a1, a2, components):
    print("Running Random Forest Holdout Analysis...")
    X_train, y_train = make_data(nodefile, vecfile)
    X_test, y_test = make_data(holdnodefile, holdvecfile)
    #initialize pca and random forest regressor
    pca = PCA(n_components=components)
    rf = RandomForestRegressor()
    # fit and transform data
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.fit_transform(X_test)
    #fit to models
    rf.fit(X_train_pca, y_train)
    #make predictions on test data
    rf_pred = rf.predict(X_test_pca)
    #get scores
    rf_score = mean_squared_error(y_test, rf_pred, squared=False)
    print("random forest score: ", rf_score)

    return rf_score

def runSVR(nodefile, vecfile, analysisfile, a1, a2):
    X, y = make_data(nodefile, vecfile)
    start = time.time() #beginning time
    regressor = SVR(kernel = 'rbf')
    n_scores = cross_val_score(regressor, X, y, scoring='neg_root_mean_squared_error')
    end = time.time()

    with open(analysisfile, 'a') as f:
        out = str(a1) + "," + str(a2) + ","
        out += str(mean(n_scores)) + "," + str(std(n_scores)) + ","
        out += vecfile + "\n"
        f.write(out)
    print('a1, a2 =', a1, ',', a2, ' MSE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)), "time: ", end-start)
    return mean(n_scores), std(n_scores)

def holdoutSVR(nodefile, vecfile, holdnodefile, holdvecfile, analysisfile, a1, a2, components):
    print("Running SVR Holdout Analysis...")
    X_train, y_train = make_data(nodefile, vecfile)
    X_test, y_test = make_data(holdnodefile, holdvecfile)
    # intialize pca and svr,dummy regression models
    # from https://towardsdatascience.com/principal-component-analysis-for-dimensionality-reduction-115a3d157bad
    pca = PCA(n_components=components)
    svr = SVR(kernel = 'rbf')
    # fit and transform data
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.fit_transform(X_test)
    #fit to models
    svr.fit(X_train_pca, y_train)
    #make predictions on test data
    svr_pred = svr.predict(X_test_pca)
    #get scores
    svr_score = mean_squared_error(y_test, svr_pred, squared=False)
    print("svr score: ", svr_score)
    return svr_score

def holdoutDummy(nodefile, vecfile, holdnodefile, holdvecfile, analysisfile, a1, a2, components):
    print("Running Dummy Holdout Analysis...")
    X_train, y_train = make_data(nodefile, vecfile)
    X_test, y_test = make_data(holdnodefile, holdvecfile)
    # intialize pca and knn,dummy regression models
    # from https://towardsdatascience.com/principal-component-analysis-for-dimensionality-reduction-115a3d157bad
    pca = PCA(n_components=components)
    dummy = DummyRegressor(strategy="median")
    # fit and transform data
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.fit_transform(X_test)
    #fit to models
    dummy.fit(X_train_pca, y_train)
    #make predictions on test data
    dummy_pred = dummy.predict(X_test_pca)
    #get scores
    dummy_score = mean_squared_error(y_test, dummy_pred, squared=False)
    print("dummy score: ", dummy_score)
    return dummy_score

def runDummy(nodefile, vecfile, analysisfile, a1, a2):
    start = time.time() #beginning time
    X, y = make_data(nodefile, vecfile)
    dummy_regr = DummyRegressor(strategy="median")
    n_scores = cross_val_score(dummy_regr, X, y, scoring='neg_root_mean_squared_error')
    with open(analysisfile, 'a') as f:
        out = str(a1) + "," + str(a2) + ","
        out += str(mean(n_scores)) + "," + str(std(n_scores)) + ","
        out += vecfile + "\n"
        f.write(out)
    end = time.time()
    print('MSE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)), "time: ", end-start)
    return mean(n_scores), std(n_scores)

def make_data(nodefile, vecfile):
    cleanVecFile = clean_vectors(vecfile)
    Xvectors = np.loadtxt(cleanVecFile, delimiter=',')
    ranksLst = np.loadtxt(nodefile, delimiter='; ', skiprows=1, usecols=4)
    yranks = np.array(ranksLst)
    return Xvectors, yranks

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
    result = stats.pearsonr(ranksArr, sortedRanksArr)#this compares ranks to ranks sorted by components
    #result = stats.pearsonr(ranksArr, components) #this compares ranks to components
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
