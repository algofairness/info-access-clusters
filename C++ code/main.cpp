// Information spreads from source nodes (seeds), over the network (IC model)
// Writes out information access vectors for each node to a file

//configuration - libraries it include
//generate make file

//#include <iostream> //#include <string> //#include <fstream> //#include <vector>
#include <iostream>     // std::cout
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <map>
#include <string>
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
bool makeNodeMap(map<string, string> &argMap, string);

map<string, string> nodeMap;

int main(int argc, char* argv[]) {
      clock_t tStart = clock();
      //command line argument implementation from http://www.cplusplus.com/articles/DEN36Up4/
      //argv[1] = srcEdges - edgelist data file
      //argv[2] = dstVectorFile file to write vectors to
      //argv[3] = alpha1 - probability of transmission between nodes in different phd programs
      //argv[4] = alpha2 - probability of transmission between nodes in same phd programs
      //argv[5] = repNumber - number of simulation repititions
      //argv[6] = simSeeds - whether to simulate all nodes as seeds
      //argv[7] = srcNodes - nodelist data file

      // Loads data in the graph
      string edgeFile = argv[1];
      // cout << edgeFile;
      string nodeFile = argv[7];

      // Determines where vectors will be saved
      //string outName;
      //cout << "Enter file path to save vectors: ";
      //cin >> outName;
      string outFileName = argv[2];

      // Determines whether vectors will be written out column-wise
      // cout << "Is this a large network? y/n ";
      // cin >> large;
    // }
    //
    // cout << edgeFile;
    //map<string, string> nodeMap;
    bool rc = makeNodeMap(nodeMap, nodeFile);
    cout << "makeNodeMap returned: " << rc << "\n";
    // test map looks good...
    //cout << "0" << "/" << nodeMap["0"] << "\n";
    //cout << "3" << "/" << nodeMap["3"] << "\n";
    //cout << "node 2==5 (want no):" << (nodeMap["2"]==nodeMap["5"]);
    //cout << "node 2==6 (want yes):" << (nodeMap["2"]==nodeMap["6"]);

    Graph netGraph = readGraph(edgeFile);
    netGraph.printGraph(nodeMap);
    vector<int> seeds = getSeeds(edgeFile);

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
    // cout << "Enter variables: \nrep (1000), maxK (100), gap (5), alpha1 (0.1), maxAlpha1 (0.5)\n";
    //int rep, maxK, gap;
    //string probStr;
    //cout << "alpha:";


    //cin >> probStr;

    //string repStr;
    //cout << "Number of repetitions for simulation:";
    //cin >> repStr;
    int rep = stoi(argv[5]);

    int maxK;
    int gap;
    maxK = 100, gap = 10;
    //float alpha1, maxAlpha1;
    // float alpha1 = 0.1, maxAlpha1 = 0.1;
    //cin >> rep >> maxK >> gap >> redo >> alpha1 >> maxAlpha1;
    bool weightExp = false;//true;

    float alpha1 = stof(argv[3]);
    float maxAlpha1 = alpha1;
    float alpha2 = stof(argv[4]);
    float maxAlpha2 = alpha2;

    string useAllSeeds = argv[6];
    //cout << "Use all seeds? y or n";
    //cin >> useAllSeeds;

    clock_t tAlph;
    for(float alpha_1 = alpha1, alpha_2 = alpha2;
              alpha_1 <= maxAlpha1 && alpha_2 <= maxAlpha2;
              alpha_1 += 0.1, alpha_2 += 0.1) {

        cout << "\n-----alpha_1 = " << alpha_1 << "-----\n";
        tAlph = clock();

        if (useAllSeeds=="yes") {
          generate_vectors(alpha_1, alpha2, rep, netGraph, nodeMap, outFileName);
        } else {
          generate_vectors_select_seeds(alpha_1, alpha_2, rep, netGraph, nodeMap, outFileName, seeds);
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
          string fromPhd = nodeMap[from];
          string toPhd = nodeMap[to];
          netGraph.addEdge(stoi(from), stoi(to), fromPhd, toPhd, dir);
        }
    }
    input.close();

    return netGraph;
}

//reads nodeFile to add phd values to a map from nodeID-> phd
bool makeNodeMap(map<string, string> &argMap, string file)
{

    ifstream inFile(file);

    if (not inFile.is_open()) return false;

    string line;

    while (getline(inFile, line))
    {
        // stream variable for parsing the line from the file
        istringstream ss(line);

        // using string for nodeId for now, but should be changed to int
        string nodeId;
        string phd;
        string skip;

        // read node, then skip dplp_id and gender, then read phd
        getline(ss, nodeId, ';');
        getline(ss, skip, ';');    // skip dplp_id
        getline(ss, skip, ';');    // skip gender
        getline(ss, phd, ';');

        //cout << "\"" << nodeId << "\"";
        //cout << ", \"" << phd << "\"";
        // add line data to map
        argMap[nodeId] = phd;
        //cout << "\n";

    }
    return true;
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
