#pragma once
#include "core.h"

class Optimizer {
public:
	unordered_set<Variable*>* variables;
	float lr;
	long long iterations;
	void zero_grad();
	virtual void step() = 0;
};


class SGD :public Optimizer {
public:
	SGD();
	SGD(unordered_set<Variable*>* variables, float lr=0.1);
	virtual void step();
};

Optimizer* get_optimizer(const string& name,
	unordered_set<Variable*>& variables);

Optimizer* get_optimizer(const string& name,
	unordered_set<Variable*>& variables, float lr = 0.1);