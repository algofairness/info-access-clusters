// Information spreads from source nodes (seeds), over the network (IC model)
// Writes out information access vectors for each node to a file

//configuration - libraries it include
//generate make file

//#include <iostream> //#include <string> //#include <fstream> //#include <vector>
#include <iostream>     // std::cout
#include <fstream>
#include <stdio.h>
#include <string.h>
//#include <bits/stdc++.h>
#include "graph.cpp"
#include "gen_vectors.h"
#include "simulation.h"
#include "print.h"

using namespace std;

Graph readGraph(string);
vector<int> getSeeds(string);
void selectHeuristic(int, int, float, int, int, int, Graph);
void algDescription();

int main() {
    // bool google_scholar_dataset = false;
    clock_t tStart = clock();
    //
    // // Setting up variables
    // string large = "n";
    // string outFileName = "dblp_correct_vectors.txt";
    // string fileName = "../data/dblp/dblp_correct_c_style.txt";
    //
    // if (google_scholar_dataset)
    // {
    //   string fileName = "../data/data/google_scholar_c_style.txt";
    //   string outFileName = "google_scholar_vectors.txt";
    //   string large = "y";
    //
    // } else {
    //   // Reads file's Name
      string name;
      cout << "Enter path to data file: ";
      cin >> name;

      // Loads data in the graph
      string fileName = name;
      // cout << fileName;

      // Determines where vectors will be saved
      string outName;
      cout << "Enter file path to save vectors: ";
      cin >> outName;
      string outFileName = outName;

      // Determines whether vectors will be written out column-wise
      // cout << "Is this a large network? y/n ";
      // cin >> large;

    // }
    //
    // cout << fileName;
    Graph netGraph = readGraph(fileName);
    netGraph.printGraph();
    vector<int> seeds = getSeeds(fileName);

    // string centerOption = "deg"; //Chooses the center
    //cout << "Central close (cent), Max degree (deg), Min max dist (dist): ");
    //cin >> option;
    // int initSeed = pickCenter(netGraph, centerOption);
    // cout << "Center: " << initSeed << endl;

    //algDescription();
    // int alg; // Reads alg's Name
    // cout << "Enter alg's id: ";
    // cin >> alg;

    //Set Simulation Variables
    // cout << "Enter variables: \nrep (1000), maxK (100), gap (5), minAlpha (0.1), maxAlpha (0.5)\n";
    //int rep, maxK, gap;
    string probStr;
    cout << "alpha:";
    cin >> probStr;

    string repStr;
    cout << "Number of repetitions for simulation:";
    cin >> repStr;
    int rep = stoi(repStr);

    int maxK;
    int gap;
    maxK = 100, gap = 10;
    //float minAlpha, maxAlpha;
    // float minAlpha = 0.1, maxAlpha = 0.1;
    //cin >> rep >> maxK >> gap >> redo >> minAlpha >> maxAlpha;
    bool weightExp = false;//true;

    float minAlpha = stof(probStr);
    float maxAlpha = minAlpha;

    string useAllSeeds;
    cout << "Use all seeds? y or n";
    cin >> useAllSeeds;


    clock_t tAlph;
    for(float alpha = minAlpha; alpha <= maxAlpha; alpha += 0.1) {
        cout << "\n-----alpha = " << alpha << "-----\n";
        tAlph = clock();

        if (useAllSeeds=="y") {
          generate_vectors(alpha, rep, netGraph, outFileName);
        } else {
          generate_vectors_select_seeds(alpha, rep, netGraph, outFileName, seeds);
        }
        // generate_vectors(alpha, rep, netGraph, outFileName);
        // simulation(seeds, alpha, rep, netGraph);
        // printProbs(netGraph, alpha, rep, outFileName);
        // add a function to write out probabilities

        cout << "Time: " << (float)(clock() - tAlph)/CLOCKS_PER_SEC << endl;
    }
    cout << "Time: " << (float)(clock() - tStart)/CLOCKS_PER_SEC << endl;

    return 0;
}

// Reads the network from file
// Format: Number of nodes - Direction of Graph ... Source - Destination
Graph readGraph(string file) {
    ifstream input;
    input.open(file);

    int numV;
    input >> numV; // Number of Nodes
    cout << "Number of Nodes: " << numV << endl;
    Graph netGraph(numV);

    bool dir;
    input >> dir; // 0: Undirected, 1: Directed

    string from, to;
    bool isSeed = false;
    while (input >> from >> to) {
        if (from == "s") {
          isSeed = true;
        } else if (not isSeed) {
          netGraph.addEdge(stoi(from), stoi(to), dir);
        }
    }
    input.close();

    return netGraph;
}


vector<int> getSeeds(string file) {
  ifstream input;
  input.open(file);
  vector<int> seeds;

  string line;
  string s = "s";
  bool isSeed = false;
  while (input >> line)
  {
    if (isSeed) {
      int seed = stoi(line);
      seeds.push_back(seed);
    }
    else if (line.at(0) == s.at(0)) {
      isSeed = true;
    //   cout << "line 158";
    //   char * pch;
    //   int n = line.length();
    //   line = line.substr(1, n);
    //   cout << line;
    //   char charArray[n];
    //   strcpy(charArray, line.c_str());
    //   pch = strtok (charArray,"\t");
    //   while (pch != NULL) {
    //     int seed;
    //     try {
    //       seed = stoi(pch);
    //     } catch (...){
    //       cout << pch;
    //       cout << "Your data file is not in the correct format. See the example and try again.";
    //     }
    //     seeds.push_back(seed);
    //     pch = strtok (NULL, "\t");
    //   }
    }
  }
  input.close();
  return seeds;
}



//
// void selectHeuristic(int algID, int init, float alpha, int rep, int k, int gap, Graph graph) {
//     vector<float> results;
//
//     switch(algID) {
//         case 1:
//             results = random(init, alpha, rep, k, gap, graph);
//             writeOnFile(results, "random", alpha, k, gap);
//             break;
//         case 2:
//             results = max_deg(init, alpha, rep, k, gap, graph);
//             writeOnFile(results, "maxdeg", alpha, k, gap);
//             break;
//         case 3:
//             results = min_deg(init, alpha, rep, k, gap, graph);
//             writeOnFile(results, "mindeg", alpha, k, gap);
//             break;
//         case 4:
//             results = k_gonz(init, alpha, rep, k, gap, graph);
//             writeOnFile(results, "gonzalez", alpha, k, gap);
//             break;
//         case 5:
//             results = naiveMyopic(init, alpha, rep, k, gap, graph);
//             writeOnFile(results, "naivemyopic", alpha, k, gap);
//             break;
//         case 6:
//             results = myopic(init, alpha, rep, k, gap, graph);
//             writeOnFile(results, "myopic", alpha, k, gap);
//             break;
//         case 7:
//             results = naiveGreedy_Reach(init, alpha, rep, k, gap, graph, true);
//             writeOnFile(results, "naivegreedy", alpha, k, gap);
//             break;
//         case 8:
//             results = greedy_Reach(init, alpha, rep, k, gap, graph, true);
//             writeOnFile(results, "greedy", alpha, k, gap);
//             break;
//         case 9:
//             results = naiveGreedy_Reach(init, alpha, rep, k, gap, graph, false);
//             writeOnFile(results, "naivereach", alpha, k, gap);
//             break;
//         case 10:
//             results = greedy_Reach(init, alpha, rep, k, gap, graph, false);
//             writeOnFile(results, "reach", alpha, k, gap);
//     }
// }
//
// void algDescription() {
//     cout << "--- \nEnter 1 for 'Random':\n Randomly chooses k nodes" << endl;
//     cout << "Enter 2 for 'Max Degree':\n Picks k nodes with maximum degrees" << endl;
//     cout << "Enter 3 for 'Min Degree':\n Picks k nodes with minimum degrees" << endl;
//     cout << "Enter 4 for 'Gonzalez':\n Each time pich the furthest node from sources -- repeat" << endl;
//     cout << "Enter 5 for 'Naive Myopic':\n Runs Simulation -- Picks k min probable nodes" << endl;
//     cout << "Enter 6 for 'Myopic':\n Runs Simulation -- Picks the min probable node -- repeat" << endl;
//     cout << "Enter 7 for 'Naive Greedy':\n Runs Simulation -- Picks the k nodes that increases min probability the most" << endl;
//     cout << "Enter 8 for 'Greedy':\n Runs Simulation -- Picks the node that increases min probability the most -- repeat" << endl;
//     cout << "Enter 9 for 'Naive Reach':\n Runs Simulation -- Picks the k nodes that increases average probability the most" << endl;
//     cout << "Enter 10 for 'Reach':\n Runs Simulation -- Picks the node that increases average probability the most -- repeat" << endl;
// }