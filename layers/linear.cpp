#include "linear.h"



Linear::Linear(int out_features, const string& activation, bool use_bias,
	const string& kernel_initializer,
	const string& bias_initializer,
	const Shape& input_shape, const vector<Layer*>& in_bounds,
	const vector<Layer*>& out_bounds,
	Variable* input_data, Variable* data, const Shape& shape,
	const vector<Variable*>& variables,
	const string& name) {
	this->out_features = out_features;
	this->activator = activation == "null" ? NULL : get_activator(activation);
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
	this->className = "Linear";
}


void Linear::initial_params(const Shape& input_shape) {
	this->input_shape = input_shape;
	this->initial_params();
}

void Linear::initial_params() {
	Variable* w = (*this->kernel_initializer)(
		Shape({ this->input_shape[0], this->out_features }));
	w->requires_grad_(true);
	this->variables.push_back(w);
	if (this->kernel_initializer != NULL) {
		delete this->kernel_initializer;
		this->kernel_initializer = NULL;
	}
	if (this->use_bias) {
		Variable* b = (*this->bias_initializer)(
			Shape({ 1, this->out_features }));
		b->requires_grad_(true);
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


Variable* Linear::operator()(Variable* inbound) {
	if (this->variables.size() == 0) {
		this->initial_params(inbound->shape.slice(1));
	}
	if (this->use_bias) {
		return dense(inbound, this->variables[0], this->variables[1], 
			GlobalGraph::IS_TRAINING);
	}
	return dense(inbound, this->variables[0], GlobalGraph::IS_TRAINING);
}


Variable* Linear::forward() {
	Variable* weight = this->variables[0];
	if(this->use_bias) {
		Variable* bias = this->variables[1];
		this->data = dense(this->input_data, weight, bias, GlobalGraph::IS_TRAINING);
	}
	else {
		this->data = dense(this->input_data, weight, GlobalGraph::IS_TRAINING);
	}
	this->feed_variable_to_next_layer(this->data);
	return this->data;
}
