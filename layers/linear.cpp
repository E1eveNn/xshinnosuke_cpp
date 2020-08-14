#include "linear.h"
#include "functional.h"


Linear::Linear(int out_features, const string& activation, bool use_bias,
	const string& kernel_initializer,
	const string& bias_initializer,
	const Shape& input_shape, const vector<Layer*>& in_bounds,
	const vector<Layer*>& out_bounds,
	Variable* input_data, Variable* data, const Shape& shape,
	const vector<Variable*>& variables,
	const string& name) {
	this->out_features = out_features;
	// this->activator = activation == "null" ? NULL : get_activator(activation);
	this->use_bias = use_bias;
	this->kernel_initializer = get_initializer(kernel_initializer);
	this->bias_initializer = bias_initializer == "null" ? NULL : get_initializer(bias_initializer);
	this->input_shape = input_shape;
	this->shape = shape;
	this->in_bounds = in_bounds;
	this->out_bounds = out_bounds;
	this->input_data = input_data;
	this->data = data;
	this->variables = variables;
	this->name = name;
}


void Linear::initial_params(Shape& input_shape) {
	this->input_shape = input_shape;
	Variable* w = (*this->kernel_initializer)(Shape({ input_shape[1], this->out_features }));
	this->variables.push_back(w);
	if (this->kernel_initializer != NULL) {
		delete this->kernel_initializer;
		this->kernel_initializer = NULL;
	}
	if (this->use_bias) {;
	Variable* b = (*this->bias_initializer)(Shape({ 1, this->out_features }));
		this->variables.push_back(b);
		if (this->bias_initializer != NULL) {
			delete this->bias_initializer;
			this->bias_initializer = NULL;
		}
	}
}

Shape Linear::compute_output_shape(Shape& input_shape) {
	return Shape({ this->out_features });
}


Variable* Linear::operator()(Variable* inbound, bool is_training) {
	if (this->variables.size() == 0) {
		this->initial_params(inbound->shape);
	}
	if (this->use_bias) {
		return dense(inbound, this->variables[0], this->variables[1], is_training);
	}
	return dense(inbound, this->variables[0], is_training);
}



Variable* Linear::forward(bool is_training) {
	Variable* weight = this->variables[0];
	if(this->use_bias) {
		Variable* bias = this->variables[1];
		this->data = dense(this->input_data, weight, bias, is_training);
	}
	else {
		this->data = dense(this->input_data, weight, is_training);
	}
	this->feed_variable_to_next_layer(this->data);
	return this->data;
}
