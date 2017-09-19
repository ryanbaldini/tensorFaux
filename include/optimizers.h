#ifndef OPTIMIZERS
#define OPTIMIZERS

#include <string>
#include <vector>
#include "Node.h"

//note that it's okay for multiple .cpp files to call this header, because everything is defined within a class
//linkers allow for classes to be defined multiple times per program (but only once per translation unit, i.e. once per .cpp)

//but might want to separating some of the function into a cpp anyway?

struct Optimizer
{
	string type;
	vector< Node* > nodes;
	
	virtual void updateNodeParameters(Node* node, int element) { return; }
	
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
			int nParameters = node->nParameters;
			for(int i=0; i<nParameters; i++) node->parameters[i] -= epsilon*(node->parameterGradient[i]);
		}
	
		virtual void compile(vector< Node* > nodes_)
		{
			nodes = nodes_;
		}
	
		SGD(): Optimizer(), epsilon(0.01) { type = "SGD";}
		SGD(double epsilon_): Optimizer(), epsilon(epsilon_) {type = "SGD";}
	};
	
	struct SGDMomentum : Optimizer
	{
		double epsilon;
		double momentum;
		vector< double* > grad;
	
		virtual void updateNodeParameters(Node* node, int element)
		{
			int nParameters = node->nParameters;
			for(int i=0; i<nParameters; i++)
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
				int nParameters = nodes[i]->nParameters;
				double* p = new double[nParameters];
				for(int j=0; j<nParameters; j++) p[j] = 0.0;
				grad.push_back(p);
			}
		}
	
		SGDMomentum(): Optimizer(), epsilon(0.01), momentum(0.99) {type = "SGDMomentum";}
		SGDMomentum(double epsilon_): Optimizer(), epsilon(epsilon_), momentum(0.99) {type = "SGDMomentum";}
		SGDMomentum(double epsilon_, double momentum_): Optimizer(), epsilon(epsilon_), momentum(momentum_) {type = "SGDMomentum";}
	};
}

#endif

