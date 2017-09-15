#include <vector>
#include <math.h>

using namespace std;

struct activation
{
	double (*activate)(double);
	double (*gradient)(double);
	activation(double (*activate_)(double), double (*gradient_)(double))
	{
		activate = activate_;
		gradient = gradient_;
	}
};

const activation none(
	[](double x) { return x; }, 
	[](double x) { return 1.0; }
);

const activation relu(
	[](double x) { return (x>0.0) ? x : 0.0; }, 
	[](double x) { return (x>0.0) ? 1.0 : 0.0; }
);

const activation sigmoid(
	[](double x) {
		return 1.0/(1.0 + exp(-x));
	},
	[](double x) {
		double tmp1 = exp(-x);
		double tmp2 = (1.0 + tmp1);
		return tmp1/(tmp2*tmp2);
	}
);


// activation relu;
// relu.activate =

// struct activation
// {
// 	virtual double operator() (double x);
// 	virtual double gradient(double x);
// };

// struct none: activation
// {
// 	virtual double operator() (double x);
// 	virtual double gradient(double x);
// };
//
// struct relu: activation
// {
// 	virtual double operator() (double x);
// 	virtual double gradient(double x);
// };
//
// struct sigmoid: activation
// {
// 	virtual double operator() (double x);
// 	virtual double gradient(double x);
// };