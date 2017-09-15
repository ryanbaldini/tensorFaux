#include <vector>
#include <string>
//#include <iostream>
#include "Graph.h"
// #include "Node.h"

using namespace std;

//data is fed in as a vector of pointers to doubles
//this is void; the final results of computation are stored in the terminal nodes
void Graph::compute(vector< double* > X)
{
	//reset all nodes to "not updated this round"
	int nNodes = nodes.size();
	for(int i=0; i<nNodes; i++) nodes[i].valuesUpdatedThisRound = false;
	
	//set input nodes to equal X
	//in order of creation
	//once set, we can "compute" them, which really does nothing except begin the recursive evaluation of the whole graph
	int j=0;
	for(int i=0; i<nNodes; i++)
	{
		if((nodes[i].parents).size() == 0)	//this implies it has no parents, hence is an input node
		{
			nodes[i].values = X[j];	//note: only copying a pointer here
			nodes[i].computeValues();
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
		if((nodes[i].children).size() == 0)	//no children -> terminal
		{
			output.push_back(nodes[i].values);
		}
	}
	return output;
}

void Graph::addInputNode(string name, vector<int> dim)
{
	nodes.push_back(InputNode(name, dim));
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
		if(nodes[i].name == parentNodeName)
		{
			parentNode = nodes[i];
			break;
		}
	}
	
	//add to nodes
	nodes.push_back(DenseNode(name, parentNode, nNeurons, &activate));
	
	//add to parents' children
	(parentNode->children).push_back(&(nodes[nodes.size()-1]));
}


// double* Graph::compute(double* X)
// {
// 	//open a queue of nodes to be updated
// 	vector<int> queue;
// 	//create vector specifying which nodes have been computed
// 	//(won't compute a node unless all its parents are ready)
// 	vector<bool> done; for(int i=0; i<nodes.size(); i++) done[i] = false;
// 	//find nodes whose parents are "input"
// 	for(int i=0; i<nodes.size(); i++)
// 	{
// 		if(contains(nodes[i].parents, "input"))
// 		{
// 			queue.push_back(i);
// 		}
// 	}
// 	//compute those layers, and add their children to the queue
// 	int i=0;
// 	while(i<queue.size())
// 	{
// 		currentNode = nodes[queue[i]];
// 		if(!done[queue[i]])	//if this hasn't already been computed
// 		{
// 			//check that its parents are ready
// 			if(parentsAreDone(currentNode))	//pass by reference
// 			{
// 				//compute the node
// 				nodes[queue[i]].compute();	//for nodes with input as parent, will have to create input nodes?
// 				//add children to queue
// 				addChildrenToQueue(currentNode, queue);	//pass both by reference
// 			}
// 			else	//parents aren't done: push it to the back of the node
// 			{
// 				queue.push_back(queue[i]);
// 			}
// 			//add this node to the end, skip to next in queue
// 		}
// 		i++
// 	}
// 	//thean find those layer's children
// 	//compute those layers
// 	//etc
// }
