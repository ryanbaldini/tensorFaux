#include <iostream>
#include "Graph.h"
#include <ctime>

using namespace std;

int main()
{	
	int seed = 0;
	Graph NN(seed);
	
	//define input dimension
	vector<int> dimInt(3);
	// int n=3;
	//example dimensions of an image
	dimInt[0] = 1;
	dimInt[1] = 128;
	dimInt[2] = 128;
	// dimInt.push_back(n);
	
	NN.addInputNode("input1", dimInt);
	NN.addConvolution2DNode("conv1", "input1", 8, {3, 3}, "valid", Activations::leakyRelu);
	NN.addConvolution2DNode("conv2", "conv1", 8, {3, 3}, "valid", Activations::leakyRelu);
	NN.addConvolution2DNode("conv3", "conv2", 8, {3, 3}, "valid", Activations::leakyRelu);
	NN.addConvolution2DNode("conv4", "conv3", 8, {3, 3}, "valid", Activations::leakyRelu);
	NN.addConvolution2DNode("conv5", "conv4", 8, {3, 3}, "valid", Activations::leakyRelu);
	NN.addDenseNode("dense1", "conv5", 100, Activations::leakyRelu);
	NN.addDenseNode("dense2", "dense1", 10, Activations::leakyRelu);
	NN.addDenseNode("dense3", "dense2", 1, Activations::sigmoid);
	NN.setLoss(Losses::binaryEntropy);
	
	//make optimizer
	Optimizers::SGD opt(0.01);
	// Optimizers::SGDMomentum opt(0.01);
	NN.setOptimizer(opt);
	
	//Make data
	vector< double* > X;
	int n = dimInt[0]*dimInt[1]*dimInt[2];	//total size of an input
	double* input1 = new double[n];
	// double* input2 = new double[n];
	for(int i=0; i<n; i++)
	{
		input1[i] = i % 10;
		// input2[i] = i*2;
	}
	X.push_back(input1);
	// input.push_back(input2);

	vector< double* > Y;
	double* y = new double[1];
	y[0] = 1.0;
	Y.push_back(y);	


	cout << NN.nodes[1]->values[0] << '\n';
	
	clock_t start = clock();
	for(int i=0; i<100; i++) NN.forwardSweep(X);
	cout << "Time spent in forwards sweeps: " << (clock()-start) / (double)(CLOCKS_PER_SEC / 1000) << '\n';
	
	cout << NN.nodes[1]->values[0] << '\n';

	//Do some training!
	// int N = 10;
	// vector< vector< double* > > XTrain(N);
	// vector< vector< double* > > YTrain(N);
	//
	// //all the same, why not
	// for(int i=0; i<N; i++)
	// {
	// 	XTrain[i] = X;
	// 	YTrain[i] = Y;
	// }
	//
	// int nEpochs = 100;
	// int batchSize = 100;
	// bool verbose = false;
	// //verbose increases training time by ~10%
	// //because it calculates the error on each batch, so extra computation required
	//
	// cout << "Training error before: " << NN.getError(XTrain, YTrain) << '\n';
	//
	// clock_t start = clock();
	// NN.train(XTrain, YTrain, nEpochs, batchSize, verbose);
	// cout << "Total time: " << (clock()-start) / (double)(CLOCKS_PER_SEC / 1000) << '\n';
	//
	// cout << "Training error after: " << NN.getError(XTrain, YTrain) << '\n';
	
	return 0;
}