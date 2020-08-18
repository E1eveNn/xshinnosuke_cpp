#pragma once
#include "../nn/functional.h"


class Activation:public Layer{
public:
	Activation(const string& act_name, 
		const vector<Layer*>& in_bounds = vector<Layer*>(),
		const vector<Layer*>& out_bounds = vector<Layer*>(),
		Variable* input_data = NULL, Variable* data = NULL, 
		const Shape& input_shape = Shape(),
		const Shape& shape = Shape(), 
		const vector<Variable*>& variables = vector<Variable*>(), 
		const string& name = "");
	Activation(const vector<Layer*>& in_bounds = vector<Layer*>(),
		const vector<Layer*>& out_bounds = vector<Layer*>(),
		Variable* input_data = NULL, Variable* data = NULL,
		const Shape& input_shape = Shape(),
		const Shape& shape = Shape(),
		const vector<Variable*>& variables = vector<Variable*>(),
		const string& name = "");
	virtual Variable* forward();
	virtual Variable* operator()(Variable* inbound);
private:
	Activation* activator;
};

class ReLU : public Activation {
public:
	ReLU(bool inplace = false, const vector<Layer*>& in_bounds = vector<Layer*>(),
		const vector<Layer*>& out_bounds = vector<Layer*>(),
		Variable* input_data = NULL, Variable* data = NULL,
		const Shape& input_shape = Shape(),
		const Shape& shape = Shape(),
		const vector<Variable*>& variables = vector<Variable*>(),
		const string& name = "");

	virtual Variable* forward();
	virtual Variable* operator()(Variable* inbound);
	unordered_map<string, bool> cache;
private:
	bool inplace;
};

class Sigmoid : public Activation {
public:
	Sigmoid(const vector<Layer*>& in_bounds = vector<Layer*>(),
		const vector<Layer*>& out_bounds = vector<Layer*>(),
		Variable* input_data = NULL, Variable* data = NULL,
		const Shape& input_shape = Shape(),
		const Shape& shape = Shape(),
		const vector<Variable*>& variables = vector<Variable*>(),
		const string& name = "");

	virtual Variable* forward();
	virtual Variable* operator()(Variable* inbound);
	unordered_map<string, bool> cache;

};

Activation* get_activator(const string& name);