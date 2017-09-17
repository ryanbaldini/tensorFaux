#include <vector>
#include <string>
#include <iostream>
#include "Node.h"

using namespace std;

//empty virtual function bases
void Node::computeMyValues() {return;}
void Node::incrementGradOnParameters() {return;}
void Node::incrementGradOnParents() {return;}
void Node::printParameters() {return;}

void Node::computeValues() 
{
	//first, check that parents' gradients have been computed
	int nParents = parents.size();
	for(int i=0; i<nParents; i++)
	{
		if(!(parents[i]->valuesUpdatedThisRound)) return;
	}
	
	computeMyValues();
	
	valuesUpdatedThisRound = true;
	
	//go to children
	int nChildren = children.size();
	for(int i=0; i<nChildren; i++) (children[i])->computeValues();	
}

void Node::incrementGradient() 
{ 
	//first, check that childrens' gradients have been computed
	int nChildren = children.size();
	for(int i=0; i<nChildren; i++)
	{
		if(!(children[i]->gradientUpdatedThisRound)) return;
	}
	
	incrementGradOnParameters();
	
	incrementGradOnParents();
	
	gradientUpdatedThisRound = true;
	
	//go to parents
	int nParents = parents.size();
	for(int i=0; i<nParents; i++) (parents[i])->incrementGradient();
}

InputNode::InputNode(string name_, vector<int> dim_)
{
	name = name_;
	dim = dim_;
	int totalDim = multVec(dim);
	values = new double[totalDim];
	gradient = new double[totalDim];
	valuesUpdatedThisRound = false;
	gradientUpdatedThisRound = false;
}
	
void InputNode::computeMyValues()
{
	return;
}
void InputNode::incrementGradOnParameters()
{
	return;
}
void InputNode::incrementGradOnParents()
{
	return;
}
void InputNode::printParameters()
{
	return;
}


DenseNode::DenseNode(string name_, Node* parentNode, int nNeurons, Activation activate_, Normaldev& rng)
{
	name = name_; 
	dim.push_back(nNeurons);
	parents.push_back(parentNode);
	activate = activate_;
	
	//values and gradient
	values = new double[nNeurons];
	gradient = new double[nNeurons];
	nonActivatedValues = new double[nNeurons];
	gradNonActivatedValues = new double[nNeurons];
	//parameters and gradient
	int dimIn = multVec(parentNode->dim);
	parameters = new double[nNeurons + nNeurons*dimIn];		//size of weights + biases
	parameterGradient = new double[nNeurons + nNeurons*dimIn];	//size of weights + biases
	
	//set the helper pointers to the right location
	biases = parameters;	//biases start at 0
	biases_gradient = parameterGradient;
	weights = new double*[nNeurons];		//each will point to the appropriate location 
	weights_gradient = new double*[nNeurons];
	for(int i=0; i<nNeurons; i++)
	{
		weights[i] = parameters + nNeurons + i*dimIn;
		weights_gradient[i] = parameterGradient + nNeurons + i*dimIn;
	}
	
	//initialize parameters at random
	for(int i=0; i<nNeurons; i++)
	{
		biases[i] = rng.dev();
		for(int j=0; j<dimIn; j++)
		{
			weights[i][j] = rng.dev();
		}
	}
}

void DenseNode::computeMyValues()
{
	Node* parent = parents[0];
	int dimIn = multVec(parent->dim);
	int dimOut = dim[0];	//only one
	
	//do matrix add, multiply, and activation
	for(int i=0; i<dimOut; i++)
	{
		nonActivatedValues[i] = biases[i];
		for(int j=0; j<dimIn; j++)
		{
			nonActivatedValues[i] += weights[i][j]*((parent->values)[j]);
		}
		values[i] = activate.activate(nonActivatedValues[i]);
	}
}

void DenseNode::incrementGradOnParameters()
{
	//update gradient on non-activated values
	int dimOut = dim[0];
	for(int i=0; i<dimOut; i++)
	{
		gradNonActivatedValues[i] = gradient[i]*activate.gradient(nonActivatedValues[i]);
	}
	
	//gradient on node's parameters depends on its parents' values
	//dense layer has only one parent
	Node* parent = parents[0];
	int dimIn = multVec(parent->dim);
	for(int i=0; i<dimOut; i++)
	{
		//double tmp = gradient[i]*gradNonActivatedValues[i];
		//consider checking if tmp is zero: might save time if we initialize these to 0 upfront? esp for those multiplications for the weights
		biases_gradient[i] += gradNonActivatedValues[i];
		for(int j=0; j<dimIn; j++)
		{
			weights_gradient[i][j] += gradNonActivatedValues[i]*(parent->values[j]);
		}
	}
}

void DenseNode::incrementGradOnParents()
{
	int dimOut = dim[0];	
	
	//dense layer has only one parent
	Node* parent = parents[0];
	int dimIn = multVec(parent->dim);
	//remember, parent's gradient might have already been incremented by another child. so only increment here.
	for(int i=0; i<dimIn; i++)
	{
		for(int j=0; j<dimOut; j++)
		{
			parent->gradient[i] += gradNonActivatedValues[j]*weights[j][i];	//bad stride on weights, but hard to fix that without hurting something else
		}
	}	
}

void DenseNode::printParameters()
{
	int dimOut = dim[0];	
	Node* parent = parents[0];
	int dimIn = multVec(parent->dim);
	
	cout << "biases:" << '\n';
	for(int i=0; i<dimOut; i++) cout << biases[i] << " , ";
	cout << '\n';
	
	cout << "weights:" << '\n';
	for(int i=0; i<dimOut; i++)
	{
		for(int j=0; j<dimIn; j++) cout << weights[i][j] << " , ";
		cout << '\n';
	}
}
