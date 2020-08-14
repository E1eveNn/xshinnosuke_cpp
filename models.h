#pragma once
#include "core.h"


class Base {
public:
	Base();
	virtual Variable* forward(Variable* x, bool is_training = true) = 0;
	void train();
	void eval();
protected:
	bool is_training;
};


class Module : public Base {
public:
	Module();
	Variable* operator()(Variable* x);
	vector<Variable*>& parameters();


protected:
	vector<Variable*> variables;
	vector<Variable*> trainable_variables;

	void collect_variables(Variable* x);
};