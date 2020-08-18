#pragma once
#include "activators.h"
#include "../nn/initializers.h"
#include "../nn/functional.h"


class Linear : public Layer {
public:
	Linear(int out_features, const string& activation = "null",
		bool use_bias = false, const string& kernel_initializer = "random",
		const string& bias_initializer = "zeros",
		const Shape& input_shape = Shape(), const vector<Layer*>& in_bounds = vector<Layer*>(),
		const vector<Layer*>& out_bounds = vector<Layer*>(),
		Variable* input_data = NULL, Variable* data = NULL,
		const Shape& shape = Shape(),
		const vector<Variable*>& variables = vector<Variable*>(),
		const string& name = "");

	virtual void initial_params(const Shape& input_shape);
	virtual void initial_params();
	Shape compute_output_shape(Shape& input_shape);
	virtual Variable* forward();
	virtual Variable* operator()(Variable* inbound);
	int out_features;
	bool use_bias;

	Activation* activator;
	Initializer* kernel_initializer;
	Initializer* bias_initializer;
};