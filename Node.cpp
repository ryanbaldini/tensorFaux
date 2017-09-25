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
	for(int i=0; i<nParameters; i++) parameters[i] = rng.dev();
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


Convolution2DNode::Convolution2DNode(string name_, Node* parentNode, int nKernels, vector<int> dimKernel, string borderMode_, Activation activate_, Normaldev& rng)
{
	//first check that input node is correct dimension
	try { 
		if(parentNode->dim.size() != 2 & parentNode->dim.size() != 3) throw string("Input is not either 2 or 3 dimensions."); 
		if(borderMode_ != "valid" & borderMode_ != "same") throw string("borderMode must be 'valid' or 'same'");
	}
	catch(string exception) { cerr << "ERROR: " << exception << '\n' ; return; }

	name = name_;
	dim.resize(3);
	parents.push_back(parentNode);

	int inputDepth, inputHeight, inputWidth;
	if(parentNode->dim.size() == 2)
	{
		inputDepth = 1; inputHeight = parentNode->dim[0]; inputWidth = parentNode->dim[1];
	}
	else
	{
		inputDepth = parentNode->dim[0]; inputHeight = parentNode->dim[1]; inputWidth = parentNode->dim[2];
	}
	//dim depends on type
	if(borderMode_ == "valid")
	{
		borderMode = "valid";

		try
		{
			if(inputHeight < dimKernel[0] | inputWidth < dimKernel[1])
			{
				throw string("Kernel doesn't fit in image. "
					"Image has dim (" + to_string(inputHeight) + "," + to_string(inputWidth) + "). " +
					"Kernel has dim (" + to_string(dimKernel[0]) + "," + to_string(dimKernel[1]) + ").");
			}
		}
		catch(string exception) { cerr << "ERROR: " << exception << '\n' ; return; }

		dim[0] = nKernels;
		dim[1] = inputHeight - dimKernel[0] + 1;
		dim[2] = inputWidth - dimKernel[1] + 1; ;

	}
	else if(borderMode_ == "same")
	{
		borderMode = "same";

		dim[0] = nKernels;
		dim[1] = inputHeight;
		dim[2] = inputWidth;

	}
	nValues = dim[0]*dim[1]*dim[2];

	values = new double[nValues];
	gradient = new double[nValues];
	nonActivatedValues = new double[nValues];
	gradNonActivatedValues = new double[nValues];

	nParameters = nKernels + inputDepth*dimKernel[0]*dimKernel[1];
	parameters = new double[nParameters];		//size of weights + biases
	parameterGradient = new double[nParameters];	//size of weights + biases

	activate = activate_;

	//set the helper pointers to the right location
	biases = parameters;	//biases start at 0
	biases_gradient = parameterGradient;
	kernels = new double***[nKernels];		//each will point to the appropriate location
	kernels_gradient = new double***[nKernels];
	for(int i=0; i<nKernels; i++)
	{
		kernels[i] = new double**[inputDepth];
		kernels_gradient[i] = new double**[inputDepth];
		for(int j=0; j<inputDepth; j++)
		{
			kernels[i][j] = new double*[inputHeight];
			kernels_gradient[i][j] = new double*[inputHeight];
			for(int k=0; k<inputHeight; k++)
			{
				kernels[i][j][k] = parameters + nKernels + dimKernel[1]*(dimKernel[0]*(inputDepth*i + j) + k);
				kernels[i][j][k] = parameterGradient + nKernels + dimKernel[1]*(dimKernel[0]*(inputDepth*i + j) + k);
			}
		}
	}

	//initialize parameters at random
	for(int i=0; i<nParameters; i++) parameters[i] = rng.dev();
}
