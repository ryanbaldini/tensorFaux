#include <iostream>

using namespace std;

struct Base
{
	int x;
	int y;
	
	virtual void subfunction1() {return;}
	virtual void subfunction2() {return;}
	
	void bigFunction()
	{
		subfunction1();
		subfunction2();
	}
	
	Base(): x(0), y(0) {}
};


struct Derived: Base
{
	void subfunction1()
	{
		x=1;
	}
	
	void subfunction2()
	{
		y=1;
	}
	
};


struct Derived2: Base
{
	void subfunction1()
	{
		x=2;
	}
	
	void subfunction2()
	{
		y=2;
	}
	
};

int main()
{
	Derived2 a;
	Base* p = &a;
	
	p->bigFunction();
	
	cout << p->x << '\n';
	cout << p->y << '\n';
	
}