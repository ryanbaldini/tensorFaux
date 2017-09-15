// #include "activations.h"
//
// // double activation::operator() (double x) { return 0.0; }
// // double activation::gradient(double x) { return 0.0; }
//
// activation::activation(double (*activate_)(double), double (*gradient_)(double))
// {
// 	activate = activate_;
// 	gradient = gradient_;
// }
//
// activation relu(
// 	[](double x) { return (x>0.0) ? x : 0.0; },
// 	[](double x) { return (x>0.0) ? 1.0 : 0.0; }
// );

// double (*activate)(double) = [](double x) { return (x>0.0) ? x : 0.0; } ;


// double hi(double x)
// {
// 	return x*x;
// }
//
// double bye(double x)
// {
// 	return x*x*x;
// }

// struct activation
// {
// 	double (*activate)(double);
// 	double (*gradient)(double);
// };

// int x(5);

// int butt(int y)
// {
// 	return y*x;
// }

// activation relu;
// double (*activate)(double) = hi;
// activate = bye;
// activate = hi;
// relu.activate = [](double) { return (x>0.0) ? x : 0.0; } ;
// relu.gradient = [](double) { return (x>0.0) ? 1.0 : 0.0; } ;
// relu.(*aci)

// double none::operator() (double x)
// {
// 	return x;
// }
// double none::gradient(double x)
// {
// 	return 1.0;
// }

// double relu::operator() (double x)
// {
// 	return (x>0.0) ? x : 0.0;
// }
// double relu::gradient(double x)
// {
// 	return (x>0.0) ? 1.0 : 0.0;
// }
//
// double sigmoid::operator() (double x)
// {
// 	return 1.0/(1.0 + exp(-x));
// }
// double sigmoid::gradient(double x)
// {
// 	double tmp1 = exp(-x);
// 	double tmp2 = (1.0 + tmp1);
// 	return tmp1/(tmp2*tmp2);
// }
//
