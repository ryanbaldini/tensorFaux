#include <vector>
#include <string>
#include "Node.h"
#include <iostream>
// #include "functions.cpp"
//#include "activations.cpp"

using namespace std;

//empty virtual function bases
void Node::computeMyValues() {return;}
void Node::computeGradOnParameters() {return;}
void Node::computeGradOnParents() {return;}
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

void Node::computeGradient() 
{ 
	//first, check that childrens' gradients have been computed
	int nChildren = children.size();
	for(int i=0; i<nChildren; i++)
	{
		if(!(children[i]->gradientUpdatedThisRound)) return;
	}
	
	computeGradOnParameters();
	
	computeGradOnParents();
	
	gradientUpdatedThisRound = true;
	
	//go to parents
	int nParents = parents.size();
	for(int i=0; i<nParents; i++) (parents[i])->computeGradient();
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
	
// void InputNode::computeValues()
// {
// 	valuesUpdatedThisRound = true;
//
// 	//compute children
// 	int nChildren = children.size();
// 	for(int i=0; i<nChildren; i++) children[i]->computeValues();
// }

void InputNode::computeMyValues()
{
	return;
}
void InputNode::computeGradOnParameters()
{
	return;
}
void InputNode::computeGradOnParents()
{
	return;
}
void InputNode::printParameters()
{
	return;
}

// void InputNode::computeGradient()
// {
// 	gradientUpdatedThisRound = true;
// }

DenseNode::DenseNode(string name_, Node* parentNode, int nNeurons, Activation activate_, Normaldev& rng)
{
	name = name_; 
	dim.push_back(nNeurons);
	parents.push_back(parentNode);
	activate = activate_;
	//values and gradient
	values = new double[dim[0]];
	gradient = new double[dim[0]];
	nonActivatedValues = new double[dim[0]];
	gradNonActivatedValues = new double[dim[0]];
	// for(int i=0; i<dim[0]; i++)
	// {
	// 	values[i] = 0.0;
	// 	gradient[i] = 0.0;
	// 	nonActivatedValues[i] = 0.0;
	// 	gradNonActivatedValues[i] = 0.0;
	// }
	//parameters and gradient
	biases = new double[dim[0]];
	biases_gradient = new double[dim[0]];
	int dimIn = multVec(parentNode->dim);
	weights = new double*[dim[0]];
	weights_gradient = new double*[dim[0]];
	for(int i=0; i<dim[0]; i++)
	{
		weights[i] = new double[dimIn];
		weights_gradient[i] = new double[dimIn];
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
	
	valuesUpdatedThisRound = false;
	gradientUpdatedThisRound = false;
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
	
// void DenseNode::computeValues()
// {
// 	//first, check that parent has been computed
// 	//maybe not necessary, since dense layers have only one parent and can only be called after the parent!
// 	// Node* parent = parents[0];	//must only be one parent
// 	// if(!(parent->valuesUpdatedThisRound)) return;
// 	Node* parent = parents[0];
//
// 	int dimIn = multVec(parent->dim);
// 	int dimOut = dim[0];	//only one
//
// 	//do matrix add, multiply, and activation
// 	for(int i=0; i<dimOut; i++)
// 	{
// 		nonActivatedValues[i] = biases[i];
// 		for(int j=0; j<dimIn; j++)
// 		{
// 			nonActivatedValues[i] += weights[i][j]*((parent->values)[j]);
// 		}
// 		values[i] = activate.activate(nonActivatedValues[i]);
// 	}
//
// 	valuesUpdatedThisRound = true;
//
// 	//compute children
// 	int nChildren = children.size();
// 	for(int i=0; i<nChildren; i++) children[i]->computeValues();
// }

void DenseNode::computeGradOnParameters()
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
		biases_gradient[i] = gradNonActivatedValues[i];
		for(int j=0; j<dimIn; j++)
		{
			weights_gradient[i][j] = gradNonActivatedValues[i]*(parent->values[j]);
		}
	}
}

void DenseNode::computeGradOnParents()
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
