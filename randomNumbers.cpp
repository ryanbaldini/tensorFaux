#include <math.h>
#include <randomNumbers.h>
#include <vector>

Ranq1::Ranq1(unsigned long long j) : v(4101842887655102017LL) 
{
	v ^= j;
	v = int64();
}

inline unsigned long long Ranq1::int64()
{
	v ^= v >> 21; v ^= v << 35; v ^= v >> 4;
	return v * 2685821657736338717LL;
}

inline double Ranq1::doub() { return 5.42101086242752217E-20 * int64();}
inline unsigned Ranq1::int32() { return (unsigned int)int64(); }

Normaldev::Normaldev(double mmu, double ssig, unsigned long long i) : Ranq1(i), mu(mmu), sig(ssig) {}
	
double Normaldev::dev()
{
	double u, v, x, y, q;
	do
	{
		u = doub();
		v = 1.7156*(doub() - 0.5);
		x = u - 0.449871;
		double v_abs = (v > 0.0) ? v : -v;
		y = v_abs + 0.386595;
		q = x*x + y*(0.19600*y - 0.25472*x);
	}
	while(q > 0.27597 && (q > 0.27846 || v*v > -4.0*log(u)*u*u ));
	return mu + sig*v/u;
}

RNG::RNG(double mmu, double ssig, unsigned long long i): Normaldev(mmu, ssig, i) {}

void RNG::shuffle(vector<int>& x)
{
	int size = x.size();
	for(int i=0; i<size; i++)
	{
		int pos = i + (int64() % (size-i));		//select position from remaining elements
		int tmp = x[i];
		x[i] = x[pos];
		x[pos] = tmp;
	}
}