#ifndef GRAPH
#define GRAPH

#include <vector>
#include <string>
#include "Node.h"
//#include "activations.cpp"

using namespace std;

struct Graph
{
	vector<Node> nodes;		//problem: does each of these now have to be a Node, i.e. not derived? I think so.
	void compute(vector< double* > X);
	vector< double* > computeAndReturn(vector< double* > X);
	void addInputNode(string name, vector<int> dim);
	void addDenseNode(string name, string parentNodeName, int nNeurons, activation activate);
};

#endif