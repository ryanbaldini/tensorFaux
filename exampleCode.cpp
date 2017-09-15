#include "Graph.h"
#include "activations.cpp"

Graph CNN;

CNN.addInputNode(name="input1", dim = (100,100));	//adds an Input node called input1
CNN.addInputNode(name="input2", dim = 1);			//adds an Input node called input1
CNN.addConvNode(name="conv1", inputNode="input1", activation=relu(), filterDim = (2,4), nFilters = 16);		//adds conv layer called conv1, which operates on input1
CNN.addConvNode(name="conv2", inputNode="input2", activation=relu(), filterDim = (2,4), nFilters = 16);		//adds conv layer called conv2, which operates on input2
CNN.addMultNode(name="mult", inputNode1="conv1", inputNode2="conv2", activation=none());		//adds node that multiplies to other nodes, possibly with activation
//thought: mult requires that the two inputs have the same dims.
CNN.addDenseNode(name="dense1", inputNode="mult", nNeurons = 100, activation=relu());	//given data is pointer, no need for flatten?
CNN.addDenseNode(name="dense2", inputNode="dense1", nNeurons = 10, activation=relu());
CNN.addDenseNode(name="output", inputNode="dense1", nNeurons = 1, activation=sigmoid());

//Q: how do we know what is an input layer?
//A: has no parents!

//Q: how do we know what is an output layer? 
//A: has no children!

//X1 is pointer to dim = (100,100)
//X2 is pointer to dim = 1
X = (X1,X2);	//either a pointer pair or a vector of the points
CNN.compute(X);
//assume that order is in the order the inputs were created