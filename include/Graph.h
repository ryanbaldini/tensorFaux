#ifndef GRAPH
#define GRAPH

#include <vector>
#include <string>
#include "Node.h"
#include "loss.h"
#include "randomNumbers.h"

using namespace std;

struct Graph
{
	vector<Node*> nodes;
	
	Loss loss;
	Normaldev rng;
	
	void forwardSweep(vector< double* > X);
	void backwardSweep(vector< double* > Y);
	// vector< double* > computeAndReturn(vector< double* > X);
	
	void addInputNode(string name, vector<int> dim);
	void addDenseNode(string name, string parentNodeName, int nNeurons, Activation activate);
	
	void printParameters(string name);
	
	void setLoss(Loss loss_);
	
	Graph();
	Graph(unsigned long long seed);
};

#endif