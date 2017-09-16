#ifndef ACTIVATIONS
#define ACTIVATIONS

#include <string>
#include <math.h>

using namespace std;

struct Activation
{
	string type;
	double (*activate)(double);
	double (*gradient)(double);
	
	Activation(): type("none")	//default to none
	{
		activate = [](double x) { return x; };
		gradient = [](double x) { return 1.0; };
	}
	
	Activation(string type_, double (*activate_)(double), double (*gradient_)(double)): type(type_)
	{
		activate = activate_;
		gradient = gradient_;
	}
};

namespace Activations
{
	const Activation none(
		"none",
		[](double x) { return x; },
		[](double x) { return 1.0; }
	);

	const Activation relu(
		"relu",
		[](double x) { return (x>0.0) ? x : 0.0; },
		[](double x) { return (x>0.0) ? 1.0 : 0.0; }
	);

	const Activation sigmoid(
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