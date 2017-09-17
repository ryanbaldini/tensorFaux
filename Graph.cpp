#include <vector>
#include <string>
#include <iostream>
#include "Graph.h"

using namespace std;

Graph::Graph(): rng(0, 0.001, 0) {}

Graph::Graph(unsigned long long seed): rng(0, 0.001, seed) {}

void Graph::printParameters(string name)
{
	int nNodes = nodes.size();
	for(int i=0; i<nNodes; i++)
	{
		if(nodes[i]->name == name)
		{
			cout << " - - - - - " << '\n';
			cout << name + " parameters:" << '\n';
			nodes[i]->printParameters();
			cout << " - - - - - " << '\n';
			return;
		}
	}
}

void Graph::setLoss(Loss loss_)
{
	loss = loss_;
}

void Graph::setOptimizer(Optimizer optimizer_)
{
	optimizer = optimizer_;
	// optimizer.nodes = nodes;
	optimizer.compile(nodes);
}

//data is fed in as a vector of pointers to doubles
void Graph::forwardSweep(vector< double* > X)
{
	//reset all nodes to "not updated this round"
	int nNodes = nodes.size();
	for(int i=0; i<nNodes; i++) nodes[i]->valuesUpdatedThisRound = false;
	
	//set input nodes to equal X
	//in order of creation
	//then compute them, starting the forward sweep
	int j=0;
	for(int i=0; i<nNodes; i++)
	{
		if((nodes[i]->parents).size() == 0)	//this implies it has no parents, hence is an input node
		{
			nodes[i]->values = X[j];	//note: only copying a pointer here
			nodes[i]->computeValues();
			j++;
		}
	}
}

void Graph::backwardSweep(vector< double* > Y)
{
	//reset all nodes to "gradient not updated this round"
	//and set value gradients to 0 (but not parameter gradients)
	int nNodes = nodes.size();
	for(int i=0; i<nNodes; i++)
	{
		nodes[i]->gradientUpdatedThisRound = false;
		int dim = multVec(nodes[i]->dim);
		for(int j=0; j<dim; j++) (nodes[i]->gradient)[j] = 0.0;
	}
	
	//get the losses on output nodes
	//in order of creation
	//then backprop gradients
	int j=0;
	for(int i=0; i<nNodes; i++)
	{
		if((nodes[i]->children).size() == 0)	//this implies it has no children, hence is an output node
		{
			//get gradient on values
			int dim = multVec(nodes[i]->dim);
			for(int k=0; k<dim; k++) (nodes[i]->gradient)[k] = loss.gradient(Y[j][k], (nodes[i]->values)[k]);	
			//start the chain
			nodes[i]->incrementGradient();
			j++;
		}
	}
}

void Graph::trainBatch(vector< vector< double* > > X, vector< vector< double* > > Y)
{
	int nSamples = X.size();
	try { if(nSamples != Y.size()) throw string("X and Y do not have the same number of samples."); }	
	catch(string exception) { cerr << "ERROR: " << exception << '\n' ; }
	
	//set parameter gradients to 0
	int nNodes = nodes.size();
	for(int i=0; i<nNodes; i++)
	{
		int dim = multVec(nodes[i]->dim);
		for(int j=0; j<dim; j++) (nodes[i]->parameterGradient)[j] = 0.0;
	}
	
	//process data
	for(int i=0; i<nSamples; i++)
	{
		forwardSweep(X[i]);
		backwardSweep(Y[i]);	//increments parameter gradients each time
	}
	
	//take mean of incremented gradients to get the overall gradient
	for(int i=0; i<nNodes; i++)
	{
		int dim = multVec(nodes[i]->dim);
		for(int j=0; j<dim; j++) (nodes[i]->parameterGradient)[j] = (nodes[i]->parameterGradient)[j]/nSamples;
	}
	
	//update parameters
	optimizer.updateParameters();
}

// vector< double* > Graph::computeAndReturn(vector< double* > X)
// {
// 	compute(X);
// 	//return pointers to values of terminal nodes
// 	int nNodes = nodes.size();
// 	vector< double* > output;
// 	for(int i=0; i<nNodes; i++)
// 	{
// 		if((nodes[i]->children).size() == 0)	//no children -> terminal
// 		{
// 			output.push_back(nodes[i]->values);
// 		}
// 	}
// 	return output;
// }

void Graph::addInputNode(string name, vector<int> dim)
{
	//check for errors
	int nNodes = nodes.size();
	try
	{
		for(int i=0; i<nNodes; i++)
		{
			if(nodes[i]->name == name) throw "The name '" + name + "' is already taken.";
		}
	}
	catch(string exception) { cerr << "ERROR: " << exception << '\n' ; }
		
	Node* newNode = new InputNode(name, dim);
	nodes.push_back(newNode);
	//no parents to deal with
}

void Graph::addDenseNode(string name, string parentNodeName, int nNeurons, Activation activate)
{
	//check for errors
	int nNodes = nodes.size();
	try
	{
		for(int i=0; i<nNodes; i++)
		{
			if(nodes[i]->name == name) throw "The name '" + name + "' is already taken.";
		}
		if(nNeurons <= 0) throw string("nNeurons must be positive integer");
	}
	catch(string exception) { cerr << "ERROR: " << exception << '\n' ; }
	
	//define parent
	//probably make this into a function
	Node* parentNode;
	for(int i=0; i<nNodes; i++)
	{
		if((nodes[i])->name == parentNodeName)
		{
			parentNode = nodes[i];
			break;
		}
	}
	
	//add to nodes
	Node* newNode = new DenseNode(name, parentNode, nNeurons, activate, rng);
	nodes.push_back(newNode);
	
	//add to parents' children
	(parentNode->children).push_back(newNode);
}
