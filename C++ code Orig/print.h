//Print on file

#ifndef print_h
#define print_h

#include <stdio.h>

// Help accessing array values from https://www.geeksforgeeks.org/pass-2d-array-parameter-c/
void writeVectorsSeedSubset(string outName, int rep, int n, float *vectors, vector<int> seeds) {
  ofstream outMin (outName);
  int num_seeds = seeds.size();
  for (int k = 0; k < num_seeds; k++) {
    outMin << seeds.at(k) << ",";
  }
  outMin << endl;
  for (int i = 0; i < n; i++) {
    // outMin << i << ",";
    for (int j = 0; j < num_seeds; j++) {
      float output = float(*((vectors+i*num_seeds) + j))/rep;
      outMin << output << ",";
    }
    outMin << endl;
  }
  outMin.close();
}

void writeVectors(string outName, int rep, int n, float *vectors) {
  ofstream outMin (outName);
  for (int i = 0; i < n; i++) {
    // outMin << i << ",";
    for (int j = 0; j < n; j++) {
      float output = float(*((vectors+i*n) + j))/rep;
      outMin << output << ",";
    }
    outMin << endl;
  }
  outMin.close();
}

void printProbs(Graph& netGraph, float alpha, int rep, string fileName) {
    // string fileName = "../Exp/Results/" + algName + "_" + to_string(k) + "_" + to_string(int(alpha*10)) + ".txt";
    ofstream outMin (fileName);
    for(int i = 0; i < netGraph.n; i++)
        outMin << float(i) << "\t"<< float(netGraph.prob[i])/rep << endl;
    outMin.close();
}

// void printProbs(Graph& netGraph, string algName, int k, float alpha, int rep) {
//     string fileName = "../Exp/Results/" + algName + "_" + to_string(k) + "_" + to_string(int(alpha*10)) + ".txt";
//     ofstream outMin (fileName);
//     for(int i = 0; i < netGraph.n; i++)
//         outMin << float(netGraph.prob[i])/rep << endl;
//     outMin.close();
// }

void writeOnFile(vector<float> results, string algName, float alpha, int k, int gap) {
    string fileName = "../Exp/Results/All_" + algName + "_" + to_string(int(alpha*10)) + "_min.txt";
    ofstream outMin (fileName);
    for(int i = 0; i <= k; i += gap)
        outMin << i << ": " << results[i/gap] << endl;
    outMin.close();
}

void writeWeighted(Graph& netGraph, string name, int alpha, int round, int rep, bool isWeighted) {
    int numV = netGraph.n;
    string fileName = "../Exp/Weight/" + name + "/" + to_string(alpha);
    if(isWeighted)
        fileName += "_weight";
    else
        fileName += "_simple";
    ofstream outPut (fileName + to_string(round) + ".txt");

    vector<bool> seen(numV, 0);
    int cand = 0;
    float maxim;
    for(int i = 0; i < numV; i++) {
        maxim = 0;
        for(int j = 0; j < numV; j++) {
            if(seen[j]) continue;
            if(netGraph.weight[j] > maxim) {
                cand = j;
                maxim = netGraph.weight[j];
            }
        }
        seen[cand] = true;
        outPut << (float) netGraph.prob[cand] / rep << "\t" << netGraph.weight[cand] << endl;
    }

    outPut.close();
}

void writeAve(Graph& netGraph, string name, int alpha, int redo, int rep) {
    int minPerc = netGraph.n;
    int ninPer = 0, eighPer = 0, sevPer = 0, sixPer = 0, fivPer = 0;
    int ninDiff = 0, eighDiff = 0, sevDiff = 0, sixDiff = 0, fivDiff = 0;

    int numV = netGraph.n;
    string fileName = "../Exp/Weight/" + name + "_" + to_string(alpha);
    ofstream outPut (fileName + ".txt");

    outPut << "For all top " << float(minPerc * 100 / numV) << "% the probability increased\n";
    outPut << "For top 90% probability increased for " << ninPer * 100 / (redo * numV / 10) << "% \n";
    outPut << "With average of: " << float(ninDiff) / (redo * rep * numV / 10) << endl;
    outPut << "For top 80% probability increased for " << eighPer * 100 / (redo * numV * 2 / 10) << "% \n";
    outPut << "With average of: " << float(eighDiff) / (redo * rep * numV * 2 / 10) << endl;
    outPut << "For top 70% probability increased for " << sevPer * 100 / (redo * numV * 3 / 10) << "% \n";
    outPut << "With average of: " << float(sevDiff) / (redo * rep * numV * 3 / 10) << endl;
    outPut << "For top 60% probability increased for " << sixPer * 100 / (redo * numV * 4 / 10) << "% \n";
    outPut << "With average of: " << float(sixDiff) / (redo * rep * numV * 4 / 10) << endl;
    outPut << "For top 50% probability increased for " << fivPer * 100 / (redo * numV * 5 / 10) << "% \n";
    outPut << "With average of: " << float(fivDiff) / (redo * rep * numV * 5 / 10) << endl;
    outPut.close();
}

void computeWeight(Graph& netGraph, vector<int> simp, vector<int> weight, int round, int eps) {
    int numV = netGraph.n;
    int diff;

    int minPerc = netGraph.n;
    int ninPer = 0, eighPer = 0, sevPer = 0, sixPer = 0, fivPer = 0;
    int ninDiff = 0, eighDiff = 0, sevDiff = 0, sixDiff = 0, fivDiff = 0;

    vector<bool> seen(numV, 0);
    int cand = 0, ctr = 0;
    float maxim;
    for(int i = 0; i < numV; i++) {
        maxim = 0;
        for(int j = 0; j < numV; j++) {
            if(seen[j]) continue;
            if(netGraph.weight[j] > maxim) {
                cand = j;
                maxim = netGraph.weight[j];
            }
        }
        seen[cand] = true;
        diff = weight[cand] - simp[cand];
        if(diff < 0)
            minPerc = min(minPerc, ctr);
        if(ctr < numV / 10) {
            ninDiff += diff;
            if(diff >= -eps)
                ninPer++;
        }
        if(ctr < numV * 2 / 10) {
            eighDiff += diff;
            if(diff >= -eps)
                eighPer++;
        }
        if(ctr < numV * 3 / 10) {
            sevDiff += diff;
            if(diff >= -eps)
                sevPer++;
        }
        if(ctr < numV * 4 / 10) {
            sixDiff += diff;
            if(diff >= -eps)
                sixPer++;
        }
        if(ctr < numV * 5 / 10) {
            fivDiff += diff;
            if(diff >= -eps)
                fivPer++;
        }else return;
        ctr++;
    }
}

#endif /* print_h */
