#ifndef RANDOM
#define RANDOM

#include <vector>

using namespace std;

struct Ranq1
{
	unsigned long long v;
	
	Ranq1(unsigned long long j);
	
	inline unsigned long long int64();
	
	inline double doub();
	inline unsigned int32();
};

struct Normaldev: Ranq1
{
	double mu, sig;
	
	Normaldev(double mmu, double ssig, unsigned long long i);
	
	double dev();
};

struct RNG: Normaldev
{
	RNG(double mmu, double ssig, unsigned long long i);
	void shuffle(vector<int>& x);
};


#endif