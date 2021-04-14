//Network Data Structure

#include <iostream>
#include <random>
#include <climits>

using namespace std;

// Adjacency List Node
struct AdjListNode {
    int id;
    string phd;
    struct AdjListNode* next;
};

// Adjacency List
struct AdjList {
    struct AdjListNode *head;
};

// Class Graph
class Graph {
public:
    int n;
    int *degree, *in_degree;
    int *prob, *group;
    struct AdjList* neighbors;
    float* weight;
    //int* thresh;

    Graph(int n) {
        this->n = n;
        degree = new int [n];
        in_degree = new int [n];
        prob = new int [n];
        weight = new float [n];
        group = new int [n];
        //thresh = new int [n];
        neighbors = new AdjList [n];
        for (int i = 0; i < n; ++i) {
            neighbors[i].head = NULL;
            degree[i] = in_degree[i] = prob[i] = 0;
            weight[i] = 1;
            group[i] = 1; //thresh[i] = 1;
        }
    }

    void setRand() {
        //srand(static_cast<unsigned int>(time(NULL)));
        random_device rand_dev;
        mt19937 generator(rand_dev());
        uniform_int_distribution<int> distrW(0, INT_MAX);
        uniform_int_distribution<int> distrG(0, 1);

        for (int i = 0; i < n; ++i) {
            weight[i] = (float) distrW(generator)/INT_MAX;
            group[i] = distrG(generator);
            uniform_int_distribution<int> distrT(0, in_degree[i]);
            //thresh[i] = (int) distrT(generator); // Thresh = 0 or d_i?
        }
    }

    // in the future, instead of phd it could be a struct of relevant attributes
    AdjListNode* newAdjListNode(int id, string phd) {
        AdjListNode* newNode = new AdjListNode;
        newNode->id = id;
        newNode->phd = phd;
        newNode->next = NULL;
        return newNode;
    }

    void addEdge(int src, int dest, string srcPhd, string destPhd, bool dir) {
        degree[src]++;
        in_degree[dest]++;
        AdjListNode* newNode = newAdjListNode(dest, destPhd);
        newNode->next = neighbors[src].head; //not sure what this does
        neighbors[src].head = newNode; //assigned head of src to be pointer to the new node
        if(dir) { return; } //if graph is directed stop here
        degree[dest]++;
        newNode = newAdjListNode(src, srcPhd);
        newNode->next = neighbors[dest].head;
        neighbors[dest].head = newNode;
    }

    void printGraph(map<string, string> map) {
        for (int v = 0; v < n; ++v) {
            AdjListNode* iter = neighbors[v].head;
            cout<<"Vertex " << v << ":";
            while (iter) {
                cout << " " << iter->id;
                iter = iter->next;
            }
            cout << endl;
        }
    }
};
