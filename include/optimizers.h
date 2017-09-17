#ifndef OPTIMIZERS
#define OPTIMIZERS

#include <vector>
#include "Node.h"
// #include "Graph.h"

struct Optimizer
{
	// Graph* graph;
	vector< Node* > nodes;
	
	virtual void updateNodeParameters(Node* node, int element) {return;}
	
	void updateParameters()
	{
		int nNodes = nodes.size();
		for(int i=0; i<nNodes; i++)
		{
			updateNodeParameters(nodes[i], i);
		}
	}

	virtual void compile(vector< Node* > nodes_)
	{
		nodes = nodes_;
	}
};

namespace Optimizers
{
	struct SGD : Optimizer
	{
		double epsilon;
	
		virtual void updateNodeParameters(Node* node, int element)
		{
			int dim = multVec(node->dim);
			for(int i=0; i<dim; i++) node->parameters[i] -= epsilon*(node->parameterGradient[i]);
		}
	
		virtual void compile(vector< Node* > nodes_)
		{
			nodes = nodes_;
		}
	
		SGD(): Optimizer(), epsilon(0.0001) {};
		SGD(double epsilon_): Optimizer(), epsilon(epsilon_) {};
	};
	
	struct SGDMomentum : Optimizer
	{
		double epsilon;
		double momentum;
		vector< double* > grad;
	
		virtual void updateNodeParameters(Node* node, int element)
		{
			int dim = multVec(node->dim);
			for(int i=0; i<dim; i++)
			{
				grad[element][i] = momentum*grad[element][i] + (1-momentum)*(node->parameterGradient[i]);
				node->parameters[i] -= epsilon*grad[element][i];
			}
		}
	
		virtual void compile(vector< Node* > nodes_)
		{
			nodes = nodes_;
			//now set up the momentum values
			int nNodes = nodes_.size();
			for(int i=0; i<nNodes; i++)
			{
				int dim = multVec(nodes[i]->dim);
				double* p = new double[dim];
				for(int j=0; j<dim; j++) p[j] = 0.0;
				grad.push_back(p);
			}
		}
	
		SGDMomentum(): Optimizer(), epsilon(0.0001), momentum(0.99) {};
		SGDMomentum(double epsilon_): Optimizer(), epsilon(epsilon_), momentum(0.99) {};
		SGDMomentum(double epsilon_, double momentum_): Optimizer(), epsilon(epsilon_), momentum(momentum_) {};
	};
	
}

#endif

//
//
// DNN.trainBatch(X, Y);
//does one update on the batch
//i.e. calculate gradient on the whole batch, and then one param update

//process: 
//clear stored gradients on parameters
//loop through data points
	//forward pass
	//backward pass
//at end of loop, ready for the optimizer to update its state
//have it loop through each node and run its update
	//e.g. SGD just adds the node's param gradient to its params; doesn't hold any additional data
	//e.g. Momentum SGD calculates the updated estimate using the node's param gradients, then updates the params
	//e.g. Adam updates estimates of first two moments, then updates nodes' params

//BUT THIS REQUIRES that params are all the same object for every node... i.e. it will ask for double* parameters!
//any way around this?

	
//this is called after each data point is processed
// void incrementOptimizerState()	//member function of an optimizer... which presumably holds a pointer to the graph itself? or something
// {
// 	for(int i=0; i<nNodes; i++)
// 	{
// 		//update various optimizer stuff
// 		//e.g.
// 	}
// }
