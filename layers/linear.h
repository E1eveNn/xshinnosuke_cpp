#pragma once
#include "core.h"
#include "activators.h"
#include "initializers.h"


class Linear : public Layer {
public:
	Linear(int out_features, const string& activation = "null", bool use_bias = true,
		const string& kernel_initializer = "random",
		const string& bias_initializer = "zeros",
		const Shape& input_shape = Shape(), const vector<Layer*>& in_bounds = vector<Layer*>(),
		const vector<Layer*>& out_bounds = vector<Layer*>(),
		Variable* input_data = NULL, Variable* data = NULL,
		const Shape& shape = Shape(), 
		const vector<Variable*>& variables = vector<Variable*>(), 
		const string& name = "");

	void initial_params(Shape& input_shape);
	Shape compute_output_shape(Shape& input_shape);
	virtual Variable* forward(bool is_training = true);
	virtual Variable* operator()(Variable* inbound, bool is_training = true);
	int out_features;
	bool use_bias;
	Initializer* kernel_initializer;
	Initializer* bias_initializer;
};