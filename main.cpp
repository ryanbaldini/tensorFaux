#include <iostream>
#include "Graph.h"
#include <ctime>

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
	DNN.addDenseNode("dense1", "input1", 100, Activations::leakyRelu);
	DNN.addDenseNode("dense2", "dense1", 100, Activations::leakyRelu);
	DNN.addDenseNode("dense3", "dense2", 100, Activations::leakyRelu);
	DNN.addDenseNode("dense4", "dense3", 10, Activations::leakyRelu);
	DNN.addDenseNode("output", "dense4", 1, Activations::sigmoid);
	DNN.setLoss(Losses::binaryEntropy);
	
	//make optimizer
	Optimizers::SGD opt(0.01);
	// Optimizers::SGDMomentum opt(0.01);
	DNN.setOptimizer(opt);
	
	//Make data
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

	vector< double* > Y;
	double* y = new double[1];
	y[0] = 1.0;
	Y.push_back(y);
	
	//Do some training!
	int N = 10000;
	vector< vector< double* > > XTrain(N);
	vector< vector< double* > > YTrain(N);
	
	//all the same, why not
	for(int i=0; i<N; i++)
	{
		XTrain[i] = input;
		YTrain[i] = Y;
	}
	
	int nEpochs = 100;
	int batchSize = 100;
	bool verbose = false;
	//verbose causes it to take like 40% longer
	
	cout << "Training error before: " << DNN.getError(XTrain, YTrain) << '\n';
	
	clock_t start = clock();
	DNN.train(XTrain, YTrain, nEpochs, batchSize, verbose);
	cout << "Training time: " << (clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << '\n';
		
	cout << "Training error after: " << DNN.getError(XTrain, YTrain) << '\n';
	
	return 0;
}