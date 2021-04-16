import numpy as np
from sklearn.decomposition import PCA
from io import StringIO

infile = "output/vectors/vectors_simTest.txt"
outfile = "output/analysis/pca.txt"

def main():
    cleanFileObj = clean_vectors(infile)
    X = np.loadtxt(cleanFileObj, delimiter=',')
    print("Matrix: \n", X, "\n")
    run_pca(X)
    return True

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

#takes as input a numpy matrix, then performs PCA analysis on it
#info on analysis: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
def run_pca(matrix):
    X = matrix
    pca1 = PCA(n_components=X.shape[0])
    pca1.fit(X)
    write_pca("pca1", pca1, outfile)

    pca2 = PCA(n_components=1)
    pca2.fit(X)
    write_pca("pca1", pca2, outfile)

    print("------------------PCA1------------------\n")
    print("components_: \n", pca1.components_, "\n")
    print("explained_variance_: \n", pca1.explained_variance_, "\n")
    print("explained_variance_ratio_: \n", pca1.explained_variance_ratio_, "\n")
    print("------------------PCA2------------------\n")
    print("components_: \n", pca2.components_, "\n")
    print("explained_variance_: \n", pca2.explained_variance_, "\n")
    print("explained_variance_ratio_: \n", pca2.explained_variance_ratio_, "\n")

    return True

def write_pca(name, pca, filename):
    with open(filename, 'a') as f:
        out = "------------------------- " + name + " -------------------------\n"
        out += "components_:\n" + str(pca.components_) + "\n"
        out += "explained_variance_:\n" + str(pca.explained_variance_) + "\n"
        out += "explained_variance_ratio_:\n" + str(pca.explained_variance_ratio_) + "\n"
        out += "\n\n"
        f.write(out)
    return True

if __name__ == '__main__':
    main()
