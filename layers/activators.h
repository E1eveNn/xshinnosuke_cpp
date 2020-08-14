#pragma once
#include "core.h"
#include "functional.h"


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
	virtual Variable* forward(bool is_training = true);
	virtual Variable* operator()(Variable* inbound, bool is_training = true);
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

	virtual Variable* forward(bool is_training = true);
	virtual Variable* operator()(Variable* inbound, bool is_training = true);
	unordered_map<string, bool> cache;
private:
	bool inplace;
};

Activation* get_activator(const string& name);