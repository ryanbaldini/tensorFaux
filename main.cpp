#include "Graph.h"
// #include "activations.cpp"
// #include "activations.h"
#include <iostream>

using namespace std;

int main()
{	
	int seed = 1;
	Graph DNN(seed);	
	vector<int> dimInt;
	int n=3;
	dimInt.push_back(n);
	
	DNN.addInputNode("input1", dimInt);
	// DNN.addInputNode("input1", dimInt);
	DNN.addDenseNode("dense1", "input1", 1, Activations::sigmoid);
	// DNN.addDenseNode("dense2", "dense1", 1, Activations::sigmoid);

	DNN.printParameters("dense1");
	// DNN.printParameters("dense2");
	
	DNN.setLoss(Losses::binaryEntropy);
	
	// cout << DNN.nodes[0]->name << '\n';
	// cout << DNN.nodes[0]->values[0] << '\n';
	// cout << (DNN.nodes[0]->children)[0]->name << '\n';
	// cout << DNN.nodes[2]->name << '\n';
	// cout << DNN.nodes[2]->values[99] << '\n';
	// cout << DNN.nodes[2]->values[100] << '\n';
	// cout << DNN.nodes[2]->parents[0]->name << '\n';
	// cout << DNN.nodes[2]->children[0]->name << '\n';
	// cout << DNN.nodes[3]->name << '\n';
	// cout << DNN.nodes[3]->values[49] << '\n';
	// cout << DNN.nodes[3]->parents[0]->name << '\n';
	// cout << DNN.nodes[3]->children[0]->name << '\n';
	// cout << DNN.nodes[4]->name << '\n';
	// cout << DNN.nodes[4]->values[0] << '\n';
	// cout << DNN.nodes[4]->parents[0]->name << '\n';
	// cout << DNN.nodes[4]->values[0] << '\n';

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
	
	cout << DNN.nodes[1]->values[0] << '\n';
	
	// DNN.addInputNode(name="input1", dim = (100,100));	//adds an Input node called input1
	// DNN.addInputNode(name="input2", dim = 1);			//adds an Input node called input1
	// DNN.addConvNode(name="conv1", inputNode="input1", activation=relu(), filterDim = (2,4), nFilters = 16);		//adds conv layer called conv1, which operates on input1
	// DNN.addConvNode(name="conv2", inputNode="input2", activation=relu(), filterDim = (2,4), nFilters = 16);		//adds conv layer called conv2, which operates on input2
	// DNN.addMultNode(name="mult", inputNode1="conv1", inputNode2="conv2", activation=none());		//adds node that multiplies to other nodes, possibly with activation
	// //thought: mult requires that the two inputs have the same dims.
	// DNN.addDenseNode(name="dense1", inputNode="mult", nNeurons = 100, activation=relu());	//given data is pointer, no need for flatten?
	// DNN.addDenseNode(name="dense2", inputNode="dense1", nNeurons = 10, activation=relu());
	// DNN.addDenseNode(name="output", inputNode="dense1", nNeurons = 1, activation=sigmoid());

	//Q: how do we know what is an input layer?
	//A: has no parents!

	//Q: how do we know what is an output layer? 
	//A: has no children!

	//X1 is pointer to dim = (100,100)
	//X2 is pointer to dim = 1
	// X = (X1,X2);	//either a pointer pair or a vector of the points
	// DNN.compute(X);
	//assume that order is in the order the inputs were created
	return 0;
}