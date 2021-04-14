#include "simulation.h"
#include "print.h"

#include <stdio.h>

using namespace std;

// function to create and write information access vectors using all nodes as seeds
void generate_vectors(float alpha1, float alpha2, int rep, Graph& graph, map<string, string> map, string outName)
{
  int n = graph.n;
  cout << to_string(n);
  // int *ary = new int[sizeX*sizeY];
  float *vectors =  new float[n*n];
  // float **vectors = new float[n][n];
  for (int i = 0; i < n; i++) {
    vector<int> seeds;
    seeds.push_back(i); //Add ith node of graph, whose id should just be i
    // cout << "Line 15" << endl;
    simulation(seeds, alpha1, alpha2, rep, graph, map);
    // cout << graph.prob[i] << endl; //prob[i] is the probability of i at this point
    for (int j = 0; j < n; j++) {
      // cout << graph.prob[j] << endl;
      vectors[j*n + i] = graph.prob[j]; //For some reason causes seg fault after 3417 iterations
      // cout << to_string(j) + "\n";
      // vectors[j][i] = graph.prob[j]; //THIS IS THE PROBLEMATIC LINE
    }
    // cout << "line 28";
  }
  // now write vectors to file
  // writeVectors(outName, rep, graph.n, vectors);
  cout << "About to start writing vectors to file" << endl;
  writeVectors(outName, rep, graph.n, (float *)vectors);

}

void generate_large_vectors(float alpha1, float alpha2, int rep, Graph& graph, map<string, string> map, string outName)
{
  // Write out information access vectors for a large dataset
  cout << "In large vector generator" << endl;
  int n = graph.n;

  ofstream outMin (outName);
  for (int i = 0; i < n; i++) {
    cout << "seed is " << to_string(i) << endl;
    vector<int> seeds;
    seeds.push_back(i); //Add ith node of graph, whose id should just be i
    simulation(seeds, alpha1, alpha2, rep, graph, map);
    //write probabilities to file
    for (int j = 0; j < n; j++) {
      outMin << graph.prob[j]/rep << ",";
    }
    outMin << endl;
  }
  outMin.close();
}

void generate_vectors_select_seeds(float alpha1, float alpha2, int rep, Graph& graph, map<string, string> map, string outName, vector<int>& all_seeds)
{
  int n = graph.n;
  cout << to_string(n);
  // int *ary = new int[sizeX*sizeY];
  int num_seeds = all_seeds.size();
  float *vectors =  new float[n*num_seeds];
  for (int i = 0; i < num_seeds; i++) {
    // cout << "Line 12" << endl;
    vector<int> seeds;
    seeds.push_back(all_seeds.at(i)); //Add ith node of graph, whose id should just be i
    // cout << "Line 15" << endl;
    simulation(seeds, alpha1, alpha2, rep, graph, map);
    // cout << graph.prob[1] << endl; //prob[i] is the probability of i at this point
    for (int j = 0; j < n; j++) {
      // cout << graph.prob[j] << endl;
      vectors[j*num_seeds + i] = graph.prob[j];
      // cout << to_string(j) + "\n";
      // vectors[j][i] = graph.prob[j]; //THIS IS THE PROBLEMATIC LINE
    }
  }
  // now write vectors to file
  writeVectorsSeedSubset(outName, rep, graph.n, (float *)vectors, all_seeds);

}
