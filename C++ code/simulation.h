// Simulate the spread of information over network
// Also outputs the results

#ifndef simulation_h
#define simulation_h

// #include <vector> //#include <time.h> //#include "graph.cpp"
// #include "computation.h"
#include <iostream>
#include <fstream>
#include <math.h>
#include <queue>
#include <stdio.h>
#include <string.h>

using namespace std;

void print_result(vector<int>&, int, int, Graph, int*);

struct simRes {
    int node;
    float minPr;
    float minWeight;
    int minGroup;
    int nodeW;
    float minPrW;
    float minWeightW;
    int minGroupW;
    int rep;
    float avePr;
};

simRes simulation(vector<int>& seeds, float alpha1, float alpha2, int rep, Graph graph, map<string, string> map) {
    //srand(static_cast<unsigned int>(time(NULL)));
    random_device rand_dev;
    mt19937 generator(rand_dev());
    uniform_int_distribution<int> distr(0, INT_MAX);

    const int numV = graph.n;
    int k = int(seeds.size());
    for(int i = 0; i < numV; i++)
        graph.prob[i] = 0;
    for(int i = 0; i < k; i++)
        graph.prob[seeds[i]] = rep;

    bool* isOn = new bool [numV];
    //vector<bool> isOn = ...
    //make sure to delete when done
    queue<int> onNodes, empty;  // Infected nodes this round, declaration of container class
    /*int* hitTime = new int [numV]{}; // Hit time of nodes for each round
     int rounds, lastNode;*/

    AdjListNode* iter = nullptr;
    // Run simulation for each repetition
    for(int simRound = 0; simRound < rep; simRound++) {
        memset(isOn, 0, numV);
        //set seed nodes to be ON
        for(int i = 0; i < k; i++)
            isOn[seeds[i]] = true;
        for(int i = 0; i < k; i++)
            onNodes.push(seeds[i]);

        /*rounds = 1;
         lastNode = onNodes.back();*/

        // Runs until no new node gets infected
        while(!onNodes.empty()) {
            //iterator?
            //what is a container?
            int curOnNode = onNodes.front();

            iter = graph.neighbors[curOnNode].head;// Neighbors of them
            while(iter) {
                float alphaVal;

                if(isOn[iter->id]) { iter = iter->next; continue; }

                if (map[to_string(curOnNode)]==iter->phd) {alphaVal = alpha2;}
                //add more else if statements here for more infection rates
                else {alphaVal = alpha1;}
                //cout << "alphaVal is: " << to_string(alphaVal) << "\n\n";
                //INT_MAX is max int
                //float is called a cast; it forces the type of distr(generator)
                //  to be a float. Static_cast is a better way to cast in C++
                if((float) distr(generator) / INT_MAX <= alphaVal) {
                    isOn[iter->id] = true;
                    graph.prob[iter->id] += 1;
                    onNodes.push(iter->id);
                    /*hitTime[iter->id] += rounds;*/
                }
                iter = iter->next;
            }
            /*if(onNodes.front() == lastNode) {
             lastNode = onNodes.back();
             rounds++;
             }*/
            onNodes.pop();
        }

        // Release memory
        swap(onNodes, empty);
    }

    int minim = 0;
    for(int v = 0; v < numV; v++)
        if(graph.prob[minim] > graph.prob[v])
            minim = v;

    int minimW = 0;
    for(int v = 0; v < numV; v++)
        if(float(graph.prob[minimW])/pow(graph.weight[minimW],2) > float(graph.prob[v])/pow(graph.weight[v],2))
            minimW = v;

    // In case "Average" is needed
    float ave = 0;
    for(int v = 0; v < numV; v++)
        ave += float(graph.prob[v] / rep);
    ave /= numV;

    float minP = float(graph.prob[minim]) / rep;
    float minW = graph.weight[minim];
    int minG = graph.group[minim];

    float minWP = float(graph.prob[minimW]) / rep;
    float minWW = graph.weight[minimW];
    int minWG = graph.group[minimW];

    simRes result = {minim, minP, minW, minG, minimW, minWP, minWW, minWG, rep, ave};

    delete[] isOn;
    delete[] iter;
    swap(onNodes, empty);
    /*delete[] hitTime;*/

    return result;
}

#endif /* simulation_h */
