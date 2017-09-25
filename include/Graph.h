#ifndef GRAPH
#define GRAPH

#include <vector>
#include <string>
#include "Node.h"
#include "loss.h"
#include "optimizers.h"
#include "randomNumbers.h"

using namespace std;

struct Graph
{
	vector<Node*> nodes;
	
	Loss loss;
	Optimizer* optimizer;
	RNG rng;
	
	void forwardSweep(vector< double* >& X);		//one vector element for each input node
	void backwardSweep(vector< double* >& Y);	//one vector element for each output node
	void trainBatch(vector< vector< double* > >& X, vector< vector< double* > >& Y, bool calculateError);	//each data point is a vector; a vector of them is batch
	void train(vector< vector< double* > >& X, vector< vector< double* > >& Y, int nEpochs = 1, int batchSize = 10, bool verbose = false);	//each data point is a vector; a vector of them is batch
	inline double getErrorOnAlreadyComputedValues(vector< double* >& Y);
	double getError(vector< double* >& X, vector< double* >& Y);
	double getError(vector< vector< double* > >& X, vector< vector< double* > >& Y);
	
	// vector< double* > computeAndReturn(vector< double* > X);
	
	void addInputNode(string name, vector<int> dim);
	void addDenseNode(string name, string parentNodeName, int nNeurons, Activation activate);
	
	void printParameters(string name);
	
	void setLoss(Loss loss_);
	
	void setOptimizer(Optimizer& optimizer_);
	
	Graph();
	Graph(unsigned long long seed);
	
};

#endif