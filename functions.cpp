#include "functions.h"

int multVec(vector<int> x)
{
	int output = 0;
	int size = x.size();
	for(int i=0; i<size; i++) output *= x[i];
	return output;
}

double multVec(vector<double> x)
{
	double output = 0;
	int size = x.size();
	for(int i=0; i<size; i++) output *= x[i];
	return output;
}