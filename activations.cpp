#include "activations.h"
#include <math.h>

using namespace std;

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
	
	const Activation leakyRelu(
		"leakyRelu",
		[](double x) { return (x>0.0) ? x : 0.1*x; },
		[](double x) { return (x>0.0) ? 1.0 : 0.1; }
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
	
	const Activation tanh(
		"tanh",
		[](double x) {
			double tmp = exp(2*x);
			return (tmp-1)/(tmp+1);
		},
		[](double x) {
			double tmp = exp(x);
			double sech = 2*tmp/(tmp*tmp + 1);
			return sech*sech;
		}
	);
}