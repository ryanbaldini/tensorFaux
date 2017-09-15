#ifndef ACTIVATIONS
#define ACTIVATIONS

#include <vector>
#include <string>
#include <math.h>

using namespace std;

struct activation
{
	string type;
	double (*activate)(double);
	double (*gradient)(double);
	
	activation(): type("none")	//default to none
	{
		activate = [](double x) { return x; };
		gradient = [](double x) { return 1.0; };
	}
	
	activation(string type_, double (*activate_)(double), double (*gradient_)(double)): type(type_)
	{
		activate = activate_;
		gradient = gradient_;
	}
};

namespace activations
{
	const activation none(
		"none",
		[](double x) { return x; },
		[](double x) { return 1.0; }
	);

	const activation relu(
		"relu",
		[](double x) { return (x>0.0) ? x : 0.0; },
		[](double x) { return (x>0.0) ? 1.0 : 0.0; }
	);

	const activation sigmoid(
		"sigmoid",
		[](double x) {
			return 1.0/(1.0 + exp(-x));
		},
		[](double x) {
			double tmp1 = exp(-x);
			double tmp2 = (1.0 + tmp1);
			return tmp1/(tmp2*tmp2);
		}
	);
}

#endif