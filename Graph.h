#ifndef GRAPH
#define GRAPH

#include <vector>
#include <string>
#include "Node.h"

using namespace std;

struct Graph
{
	vector<Node*> nodes;
	
	void compute(vector< double* > X);
	vector< double* > computeAndReturn(vector< double* > X);
	
	void addInputNode(string name, vector<int> dim);
	void addDenseNode(string name, string parentNodeName, int nNeurons, activation activate);
};

#endif