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

void Graph::setOptimizer(Optimizer& optimizer_)
{
	optimizer = &optimizer_;
	optimizer->compile(nodes);
}


//data is fed in as a vector of pointers to doubles
void Graph::forwardSweep(vector< double* >& X)
{
	//check correct number of elements
	
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

void Graph::backwardSweep(vector< double* >& Y)
{
	//check correct number of elements
	
	//reset all nodes to "gradient not updated this round"
	//and set value gradients to 0 (but not parameter gradients)
	int nNodes = nodes.size();
	for(int i=0; i<nNodes; i++)
	{
		nodes[i]->gradientUpdatedThisRound = false;
		int dim = nodes[i]->nValues;
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
			int dim = nodes[i]->nValues;
			for(int k=0; k<dim; k++) (nodes[i]->gradient)[k] = loss.gradient(Y[j][k], (nodes[i]->values)[k]);	
			//start the chain
			nodes[i]->incrementGradient();
			j++;
		}
	}
}

double Graph::getError(vector< double* >& X, vector< double* >& Y)
{		
	double error = 0.0;
	
	int nNodes = nodes.size();
	vector<int> whichNodesAreTerminal;
	for(int i=0; i<nNodes; i++)
	{
		if((nodes[i]->children).size() == 0) whichNodesAreTerminal.push_back(i);
	}
	int nTerminalNodes = whichNodesAreTerminal.size();
	
	//compute values
	forwardSweep(X);
	
	//find error
	for(int j=0; j<nTerminalNodes; j++)
	{
		int dim = nodes[whichNodesAreTerminal[j]]->nValues;
		for(int k=0; k<dim; k++) error += loss.loss(Y[j][k], nodes[whichNodesAreTerminal[j]]->values[k]);
	}
	
	return error;
}

double Graph::getError(vector< vector< double* > >& X, vector< vector< double* > >& Y)
{
	int nSamples = X.size();
	try { if(nSamples != Y.size()) throw string("X and Y do not have the same number of samples."); }
	catch(string exception) { cerr << "ERROR: " << exception << '\n' ; }
		
	double error = 0.0;
	
	for(int i=0; i<nSamples; i++)
	{
		error += getError(X[i],Y[i]);
	}
	error = error/nSamples;
	return error;
}
	
inline double Graph::getErrorOnAlreadyComputedValues(vector< double* >& Y)
{
	double error = 0.0;
	
	int nNodes = nodes.size();
	vector<int> whichNodesAreTerminal;
	for(int i=0; i<nNodes; i++)
	{
		if((nodes[i]->children).size() == 0) whichNodesAreTerminal.push_back(i);
	}
	int nTerminalNodes = whichNodesAreTerminal.size();
	
	//values already computed
	
	//find error
	for(int j=0; j<nTerminalNodes; j++)
	{
		int dim = nodes[whichNodesAreTerminal[j]]->nValues;
		for(int k=0; k<dim; k++) error += loss.loss(Y[j][k], nodes[whichNodesAreTerminal[j]]->values[k]);
	}
	
	return error;
}

void Graph::trainBatch(vector< vector< double* > >& X, vector< vector< double* > >& Y, bool verbose)
{
	int nSamples = X.size();
		
	//set parameter gradients to 0
	int nNodes = nodes.size();
	for(int i=0; i<nNodes; i++)
	{
		int nParameters = nodes[i]->nParameters;
		for(int j=0; j<nParameters; j++){ 
			(nodes[i]->parameterGradient)[j] = 0.0;
		}
	}
	
	//process data
	if(verbose)
	{
		double error = 0.0;
		for(int i=0; i<nSamples; i++)
		{
			forwardSweep(X[i]);
			error += getErrorOnAlreadyComputedValues(Y[i]);	
			backwardSweep(Y[i]);	//increments parameter gradients each time
		}
		error = error/nSamples;
		cout << "Pre-training error on batch: " << error << '\n';
		
	} else {
		for(int i=0; i<nSamples; i++)
		{
			forwardSweep(X[i]);
			backwardSweep(Y[i]);	//increments parameter gradients each time
		}	
	}	
	
	//take mean of incremented gradients to get the overall gradient
	for(int i=0; i<nNodes; i++)
	{
		int nParameters = nodes[i]->nParameters;
		for(int j=0; j<nParameters; j++)
		{
			(nodes[i]->parameterGradient)[j] = (nodes[i]->parameterGradient)[j]/nSamples;
		}
	}
	
	//update parameters
	optimizer->updateParameters();
}

void Graph::train(vector< vector< double* > >& X, vector< vector< double* > >& Y, int nEpochs, int batchSize, bool verbose)
{
	int nSamples = X.size();
	try { if(nSamples != Y.size()) throw string("X and Y do not have the same number of samples."); }	
	catch(string exception) { cerr << "ERROR: " << exception << '\n' ; return; }

	//create an order array, which will be shuffled for each epoch
	vector<int> order(nSamples);
	for(int i=0; i<nSamples; i++) order[i] = i;
	
	for(int i=0; i<nEpochs; i++)
	{
		if(verbose) cout << "***** BATCH " << i+1 << " *****" << '\n';
		
		//shuffle order in which they are processed
		rng.shuffle(order);
		
		int place = 0;
		//first do as many complete batches as possible
		{
			int nCompleteBatches = nSamples/batchSize; //rounds down
			if(nCompleteBatches > 0)
			{
				vector< vector< double* > > XBatch(batchSize);
				vector< vector< double* > > YBatch(batchSize);
				for(int j=0; j<nCompleteBatches; j++)
				{
					for(int k=0; k<batchSize; k++)
					{
						XBatch[k] = X[order[place]];
						YBatch[k] = Y[order[place]];
						place++;
					}
					trainBatch(XBatch, YBatch, verbose);
				}
			}	
		}
		//now do the remainder
		int nLeft = nSamples % batchSize;
		if(nLeft > 0)
		{
			vector< vector< double* > > XBatch(nLeft);
			vector< vector< double* > > YBatch(nLeft);
			for(int j=0; j<nLeft; j++)
			{
				XBatch[j] = X[order[place]];
				YBatch[j] = Y[order[place]];
				place++;
			}
			trainBatch(XBatch, YBatch, verbose);
		}
	}
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
	catch(string exception) { cerr << "ERROR: " << exception << '\n' ; return; }
		
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
	catch(string exception) { cerr << "ERROR: " << exception << '\n' ; return; }
	
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

void Graph::addConvolution2DNode(string name, string parentNodeName, int nKernels, vector<int> dimKernel, string borderMode, Activation activate)
{
	//check for errors
	int nNodes = nodes.size();
	try
	{
		for(int i=0; i<nNodes; i++)
		{
			if(nodes[i]->name == name) throw "The name '" + name + "' is already taken.";
		}
		if(nKernels <= 0) throw string("nKernels must be positive integer");
		if(dimKernel.size() != 2) throw string("dimKernel must have 2 elements. May enter in {x,y} format.");
		if(dimKernel[0] <= 0) throw string("dimKernel must have positive values");
		if(dimKernel[1] <= 0) throw string("dimKernel must have positive values");
		if(borderMode != "same" & borderMode != "valid") throw string("borderMode must be 'same' or 'valid'");
	}
	catch(string exception) { cerr << "ERROR: " << exception << '\n' ; return; }
	
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
	Node* newNode = new Convolution2DNode(name, parentNode, nKernels, dimKernel, borderMode, activate, rng);
	nodes.push_back(newNode);
	
	//add to parents' children
	(parentNode->children).push_back(newNode);
}
