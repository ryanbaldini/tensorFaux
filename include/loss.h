#ifndef LOSS
#define LOSS

#include <string>
#include <math.h>

using namespace std;

//remember: can define stuff within a class separately in each translation unit (cpp)
//implicitly inline, so good for small functions
struct Loss
{
	string type;
	double (*loss)(double, double);
	double (*gradient)(double, double);
	
	Loss(): type("squaredError")	//default to squared error
	{
		loss = [](double truth, double prediction) { return (prediction-truth)*(prediction-truth); };
		gradient = [](double truth, double prediction) { return 2.0*(prediction-truth); };
	}
	
	Loss(string type_, double (*loss_)(double, double), double (*gradient_)(double, double)): type(type_)
	{
		loss = loss_;
		gradient = gradient_;
	}
};

namespace Losses
{
	extern const Loss squaredError;
	extern const Loss absoluteError;
	extern const Loss binaryEntropy;
}

#endif