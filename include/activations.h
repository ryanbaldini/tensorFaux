#ifndef ACTIVATIONS
#define ACTIVATIONS

#include <string>
#include <math.h>

using namespace std;

//remember: can define stuff within a class separately in each translation unit (cpp)
//implicitly inline, so good for small functions
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
	extern const Activation none;
	extern const Activation relu;
	extern const Activation leakyRelu;
	extern const Activation sigmoid;
	extern const Activation tanh;
}

//extern const means that it's a constant that is defined in another file
//in this case, it's defined in activations.cpp
//(the fact that this works implies that extern const variables don't require immediate definition?)
//why do it this way?
//previously, I had just a activations.h file with everything defined all in one go. No externs. It worked.
//multiple .cpp files called this header, meaning each would define it.
//but since const variables have only internal linkage, this is okay: each would only look at its own copy
//but it's redundant: that code is translated multiple times
//this way works, without redundancy 

#endif