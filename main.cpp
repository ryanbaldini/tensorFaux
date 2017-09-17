#include <iostream>
#include "Graph.h"

using namespace std;

int main()
{	
	int seed = 0;
	Graph DNN(seed);	
	
	vector<int> dimInt;
	int n=3;
	dimInt.push_back(n);
	
	DNN.addInputNode("input1", dimInt);
	// DNN.addInputNode("input1", dimInt);
	DNN.addDenseNode("dense1", "input1", 3, Activations::relu);
	DNN.addDenseNode("dense2", "dense1", 1, Activations::sigmoid);
	DNN.setLoss(Losses::binaryEntropy);
	DNN.setOptimizer(Optimizers::SGD(0.0001));

	DNN.printParameters("dense2");
	// DNN.printParameters("dense2");
	
	//Do a forward pass
	vector< double* > input;
	double* input1 = new double[n];
	// double* input2 = new double[n];
	for(int i=0; i<n; i++)
	{
		input1[i] = i;
		// input2[i] = i*2;
	}
	input.push_back(input1);
	// input.push_back(input2);
	DNN.forwardSweep(input);
	
	cout << DNN.nodes[1]->values[0] << " , " << DNN.nodes[1]->values[1] << " , " << DNN.nodes[1]->values[2] << '\n';
	
	cout << DNN.nodes[2]->values[0] << '\n';
	
	//Do a backward pass
	vector< double* > Y;
	double* y = new double[1];
	y[0] = 1.0;
	Y.push_back(y);
	
	cout << DNN.nodes[2]->gradient[0] << '\n';
	DNN.backwardSweep(Y);
	cout << DNN.nodes[2]->gradient[0] << '\n';
	
	cout << DNN.nodes[1]->gradient[0] << " , " << DNN.nodes[1]->gradient[1] << " , " << DNN.nodes[1]->gradient[2] << '\n';
	
	return 0;
}