#pragma once
#include "core.h"

class Optimizer {
public:
	vector<Variable*>* variables;
	float lr;
	long long iterations;
	void zero_grad();
	virtual void step() = 0;
};


class SGD :public Optimizer {
public:
	SGD();
	SGD(vector<Variable*>* variables, float lr=0.1);
	virtual void step();
};