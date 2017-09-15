#include <vector>
#include <string>
//#include <iostream>
#include "Graph.h"

using namespace std;

//data is fed in as a vector of pointers to doubles
//this is void; the final results of computation are stored in the terminal nodes
void Graph::compute(vector< double* > X)
{
	//reset all nodes to "not updated this round"
	int nNodes = nodes.size();
	for(int i=0; i<nNodes; i++) nodes[i]->valuesUpdatedThisRound = false;
	
	//set input nodes to equal X
	//in order of creation
	//once set, we can "compute" them, which really does nothing except begin the recursive evaluation of the whole graph
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

vector< double* > Graph::computeAndReturn(vector< double* > X)
{
	compute(X);
	//return pointers to values of terminal nodes
	int nNodes = nodes.size();
	vector< double* > output;
	for(int i=0; i<nNodes; i++)
	{
		if((nodes[i]->children).size() == 0)	//no children -> terminal
		{
			output.push_back(nodes[i]->values);
		}
	}
	return output;
}

void Graph::addInputNode(string name, vector<int> dim)
{
	Node* newNode = new InputNode(name, dim);
	nodes.push_back(newNode);
	//no parents to deal with
}

void Graph::addDenseNode(string name, string parentNodeName, int nNeurons, activation activate)
{
	//define parent
	//probably make this into a function
	int nNodes = nodes.size();
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
	Node* newNode = new DenseNode(name, parentNode, nNeurons, activate);
	nodes.push_back(newNode);
	
	//add to parents' children
	(parentNode->children).push_back(newNode);
}
