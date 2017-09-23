#include <vector>
#include <string>
#include <iostream>
#include "Node.h"
#include <ctime>
#include <immintrin.h>

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
	nValues = multVec(dim);
	values = new double[nValues];
	gradient = new double[nValues];
	//is the above necessary? currently yes; these are set to 0 on backward pass; it is assumed to exist
	//but it's wasted memory and computation
	//fix?
	nParameters = 0;
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
	nValues = nNeurons;
	parents.push_back(parentNode);
	activate = activate_;
	
	//values and gradient
	values = new double[nNeurons];
	gradient = new double[nNeurons];
	nonActivatedValues = new double[nNeurons];
	gradNonActivatedValues = new double[nNeurons];
	//parameters and gradient
	int dimIn = parentNode->nValues;
	nParameters = nNeurons + nNeurons*dimIn;
	parameters = new double[nParameters];		//size of weights + biases
	parameterGradient = new double[nParameters];	//size of weights + biases
	
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
	int dimIn = parent->nValues;
	int dimOut = dim[0];	//only one
	
	//using SIMD
	//currently have to use unaligned memory; no way to ensure alignment of each 
	int nInRemainder = dimIn % 4;
	int nInInto4 = dimIn - nInRemainder;
	for(int i=0; i<dimOut; i++)
	{
		__m256d accum = _mm256_setzero_pd();
		for(int j=0; j<nInInto4; j+=4)
		{
			__m256d vWeightsI = _mm256_loadu_pd(weights[i] + j);
			__m256d vParentValues = _mm256_loadu_pd(parent->values + j);
			// accum = _mm256_fmadd_pd(vWeightsI, vParentValues, accum);	//not working for some reason
			accum += _mm256_mul_pd(vWeightsI, vParentValues);
		}
		//consider replacing the below with a single masked vector operation
		nonActivatedValues[i] = biases[i];
		nonActivatedValues[i] += accum[0];
		nonActivatedValues[i] += accum[1];
		nonActivatedValues[i] += accum[2];
		nonActivatedValues[i] += accum[3];
		for(int j = nInInto4; j<dimIn; j++) nonActivatedValues[i] += weights[i][j]*((parent->values)[j]);

		values[i] = activate.activate(nonActivatedValues[i]);
	}
}

void DenseNode::incrementGradOnParameters()
{
	int dimOut = dim[0];
	Node* parent = parents[0];
	int dimIn = parent->nValues;
	// int nOutRemainder = dimOut % 4;
	// int nOutInto4 = dimOut - nOutRemainder;
	int nInRemainder = dimIn % 4;
	int nInInto4 = dimIn - nInRemainder;	
	
	// update gradient on non-activated values
	// for(int i=0; i<dimOut; i++)
	// {
	// 	gradNonActivatedValues[i] = gradient[i]*activate.gradient(nonActivatedValues[i]);
	// }
	// for(int i=0; i<nOutInto4; i+=4)
	// {
	// 	gradNonActivatedValues[i] = gradient[i]*activate.gradient(nonActivatedValues[i]);
	// 	gradNonActivatedValues[i+1] = gradient[i+1]*activate.gradient(nonActivatedValues[i+1]);
	// 	gradNonActivatedValues[i+2] = gradient[i+2]*activate.gradient(nonActivatedValues[i+2]);
	// 	gradNonActivatedValues[i+3] = gradient[i+3]*activate.gradient(nonActivatedValues[i+3]);
	// }
	// for(int i = nOutInto4; i<dimOut; i++) gradNonActivatedValues[i] = gradient[i]*activate.gradient(nonActivatedValues[i]);
	
	//gradient on node's parameters depends on its parents' values
	//dense layer has only one parent
	for(int i=0; i<dimOut; i++)
	{
		//first gradient on non-activated values
		gradNonActivatedValues[i] = gradient[i]*activate.gradient(nonActivatedValues[i]);
		
		//now the parameters
		//biases
		biases_gradient[i] += gradNonActivatedValues[i];
		
		//weights
		// __m256d vGradNonActivatedValuesI = _mm256_set1_pd(gradNonActivatedValues[i]);
		// __m256d output;
		// for(int j=0; j<nInInto4; j+=4)
		// {
		// 	__m256d vParentValues = _mm256_loadu_pd(parent->values + j);
		// 	output = _mm256_mul_pd(vGradNonActivatedValuesI, vParentValues);
		// 	weights_gradient[i][j] += output[0];
		// 	weights_gradient[i][j+1] += output[1];
		// 	weights_gradient[i][j+2] += output[2];
		// 	weights_gradient[i][j+3] += output[3];
		// 	// weights_gradient[i][j] += gradNonActivatedValues[i]*(parent->values[j]);
		// 	// weights_gradient[i][j+1] += gradNonActivatedValues[i]*(parent->values[j+1]);
		// 	// weights_gradient[i][j+2] += gradNonActivatedValues[i]*(parent->values[j+2]);
		// 	// weights_gradient[i][j+3] += gradNonActivatedValues[i]*(parent->values[j+3]);
		// }
		// for(int j = nInInto4; j<dimIn; j++) weights_gradient[i][j] += gradNonActivatedValues[i]*(parent->values[j]);
		//failed to get any boost from vectorization
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
	int dimIn = parent->nValues;
	//remember, parent's gradient might have already been incremented by another child. so only increment here.
	//this is much better than doing the loop the other way. much better stride.
	for(int i=0; i<dimOut; i++)
	{
		for(int j=0; j<dimIn; j++)
		{
			parent->gradient[j] += gradNonActivatedValues[i]*weights[i][j];
		}
	}
	
}

void DenseNode::printParameters()
{
	int dimOut = dim[0];	
	Node* parent = parents[0];
	int dimIn = parent->nValues;
	
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
