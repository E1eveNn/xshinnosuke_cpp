#pragma once
#include "nn/core.h"


class Base {
public:
	Base();
	virtual Variable* forward(Variable* x) = 0;
	void train();
	void eval();
protected:
	vector<Variable*> variables;
	vector<Variable*> trainable_variables;
};


class Module : public Base {
public:
	Module();
	Variable* operator()(Variable* x);
	vector<Variable*>* parameters();


protected:
	void collect_variables(Variable* x);
};