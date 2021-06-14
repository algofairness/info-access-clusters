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

int main(int argc, char* argv[]) {
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
    
    // Reads file's Name
    string fileName = argv[1];

    // Determines where vectors will be saved
    string outFileName = argv[2];

    // Determines whether vectors will be written out column-wise
    // cout << "Is this a large network? y/n ";
    // cin >> large;

    // }
    
    // Loads data in the graph
    Graph netGraph = readGraph(fileName);
    //netGraph.printGraph();
    vector<int> seeds = getSeeds(fileName);

    //Set Simulation Variables
    // cout << "Enter variables: \nrep (1000), maxK (100), gap (5), minAlpha (0.1), maxAlpha (0.5)\n";
    
    // Probability of propagation (through edges)
    float alpha = stof(argv[3]);
    
    // Number of repetitions for simulation:
    int rep = stoi(argv[4]);
    
    // Use all seeds? y or n
    string useAllSeeds = argv[5];
    
    // Use multiple sources for spreading the same info (MIT)
    string multiSource = argv[6];
    if(multiSource=="y") {
        simulation(seeds, alpha, rep, netGraph);
        printProbs(netGraph, alpha, rep, outFileName);
        return 0;
    }

    if (useAllSeeds=="y") {
      generate_vectors(alpha, rep, netGraph, outFileName);
    } else {
      generate_vectors_select_seeds(alpha, rep, netGraph, outFileName, seeds);
    }
    // generate_vectors(alpha, rep, netGraph, outFileName);
    // simulation(seeds, alpha, rep, netGraph);
    // printProbs(netGraph, alpha, rep, outFileName);
    // add a function to write out probabilities

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
