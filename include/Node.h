#ifndef NODE
#define NODE

#include <vector>
#include <string>
#include "functions.h"
#include "activations.h"
#include "randomNumbers.h"

using namespace std;

struct Node
{
	string name;
	vector<int> dim;
	double* values;				//computed values of the node; altered on forward pass
	double* gradient;			//gradient on the values; altered on backward pass
	double* parameters;
	double* parameterGradient;
	vector< Node* > parents;
	vector< Node* > children;
	Activation activate;
	bool valuesUpdatedThisRound;
	bool gradientUpdatedThisRound;
	
	//these virtual functions are defined in each type of node
	void computeValues();
	virtual void computeMyValues();
	
	void incrementGradient();
	virtual void incrementGradOnParameters();
	virtual void incrementGradOnParents();
	
	virtual void printParameters();
};

struct InputNode: Node
{	
	InputNode(string name_, vector<int> dim_);	
	
	virtual void computeMyValues();	
	
	virtual void incrementGradOnParameters();
	virtual void incrementGradOnParents();
	
	virtual void printParameters();
};

struct DenseNode: Node
{
	double* nonActivatedValues;
	double* gradNonActivatedValues;
	//the following just point to the locations of the biases and weights within the node's parameter array
	//solely for convenience and readability of code 
	double* biases;
	double** weights;
	double* biases_gradient;
	double** weights_gradient;
	
	DenseNode(string name_, Node* parentNode, int nNeurons, Activation activate_, Normaldev& rng);
	
	virtual void computeMyValues();
	
	virtual void incrementGradOnParameters();
	virtual void incrementGradOnParents();
	
	virtual void printParameters();
};

#endif