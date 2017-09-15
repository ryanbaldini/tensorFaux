#ifndef NODE
#define NODE

#include <vector>
#include <string>
#include "activations.h"
#include "functions.h"

using namespace std;

struct Node
{
	string name;
	vector<int> dim;
	double* values;				//computed values of the node; altered on forward pass
	double* gradient;			//gradient on the values
	vector< Node* > parents;
	vector< Node* > children;
	activation* activate;
	bool valuesUpdatedThisRound;
	bool gradientUpdatedThisRound;
	
	//these virtual functions are defined in each type of node
	virtual void computeValues();	//no argument: get input from parents
	virtual void computeGradient();	//no argument: get input from children
};

struct InputNode: Node
{	
	InputNode(string name_, vector<int> dim_);	
	virtual void computeValues();	
	virtual void computeGradient();
};

struct DenseNode: Node
{
	double* biases;
	double** weights;
	double* biases_gradient;
	double** weights_gradient;
	DenseNode(string name_, Node* parentNode, int nNeurons, activation* activate);
	virtual void computeValues();
	virtual void computeGradient();
};

#endif