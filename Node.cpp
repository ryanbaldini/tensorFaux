#include <vector>
#include <string>
#include "Node.h"
// #include "functions.cpp"
//#include "activations.cpp"

using namespace std;

void Node::computeValues() { }
void Node::computeGradient() { }

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
	
void InputNode::computeValues()
{
	valuesUpdatedThisRound = true;
	
	//compute children
	int nChildren = children.size();
	for(int i=0; i<nChildren; i++) (children[i])->computeValues();
}
	
void InputNode::computeGradient()
{
	gradientUpdatedThisRound = true;
}

DenseNode::DenseNode(string name_, Node* parentNode, int nNeurons, activation* activate_)
{
	name = name_; 
	dim.push_back(nNeurons);
	parents.push_back(parentNode);
	activate = activate_;
	//values and gradient
	values = new double[dim[0]];
	gradient = new double[dim[0]];
	for(int i=0; i<dim[0]; i++)
	{
		values[i] = 42.0;
		gradient[i] = 0.0;
	}
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
	
	//activate = relu;
	valuesUpdatedThisRound = false;
	gradientUpdatedThisRound = false;
}
	
void DenseNode::computeValues()
{
	//first, check that parent has been computed
	//maybe not necessary, since dense layers have only one parent and can only be called after the parent!
	// Node* parent = parents[0];	//must only be one parent
	// if(!(parent->valuesUpdatedThisRound)) return;
	Node* parent = parents[0];
	
	int dimIn = multVec(parent->dim);
	int dimOut = dim[0];	//only one
	
	//do matrix add, multiply, and activation
	for(int i=0; i<dimOut; i++)
	{
		values[i] = biases[i];
		for(int j=0; j<dimIn; j++)
		{
			values[i] += weights[i][j]*((parent->values)[j]);
		}
		values[i] = activate->activate(values[i]);
	}
	
	valuesUpdatedThisRound = true;
	
	//compute children
	int nChildren = children.size();
	for(int i=0; i<nChildren; i++) (children[i])->computeValues();
}
	
void DenseNode::computeGradient()
{
	//first, check that childrens' gradients have been computed
	int nChildren = children.size();
	for(int i=0; i<nChildren; i++)
	{
		if(!(children[i]->gradientUpdatedThisRound)) return;
	}
	
	for(int i=0; i<nChildren; i++)
	{
		//increment gradient of each value based on output of this child
		
		//also increment gradient of node's specific params? (In this case, biases and weights)
	}
	
	//compute gradients of parents
	int nParents = parents.size();
	for(int i=0; i<nParents; i++) (parents[i])->computeGradient();
	
}
