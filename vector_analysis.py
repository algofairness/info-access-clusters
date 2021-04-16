import numpy as np
from sklearn.decomposition import PCA
from io import StringIO

def main():
    file = "output/vectors/vectors_experiment1-2.txt"
    cleanFileObj = clean_vectors(file)
    X = np.loadtxt(cleanFileObj, delimiter=',')
    print("Matrix: \n", X, "\n")

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

if __name__ == '__main__':
    main()
