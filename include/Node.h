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
	Activation activate;
	bool valuesUpdatedThisRound;
	bool gradientUpdatedThisRound;
	
	//these virtual functions are defined in each type of node
	void computeValues();
	virtual void computeMyValues();
	
	void computeGradient();
	virtual void computeGradOnParameters();
	virtual void computeGradOnParents();
};

struct InputNode: Node
{	
	InputNode(string name_, vector<int> dim_);	
	
	virtual void computeMyValues();	
	
	virtual void computeGradOnParameters();
	virtual void computeGradOnParents();
};

struct DenseNode: Node
{
	double* nonActivatedValues;
	double* gradNonActivatedValues;
	double* biases;
	double** weights;
	double* biases_gradient;
	double** weights_gradient;
	DenseNode(string name_, Node* parentNode, int nNeurons, Activation activate_);
	
	virtual void computeMyValues();
	
	virtual void computeGradOnParameters();
	virtual void computeGradOnParents();
};

#endif