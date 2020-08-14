#include "activators.h"

Activation::Activation(const string& act_name,
	const vector<Layer*>& in_bounds, const vector<Layer*>& out_bounds,
	Variable* input_data, Variable* data, const Shape& input_shape,
	const Shape& shape, const vector<Variable*>& variables, const string& name) {
	this->activator = get_activator(act_name);
	this->input_shape = input_shape;
	this->shape = shape;
	this->in_bounds = in_bounds;
	this->out_bounds = out_bounds;
	this->input_data = input_data;
	this->data = data;
	this->variables = variables;
	this->name = name;
}

Activation::Activation(const vector<Layer*>& in_bounds, const vector<Layer*>& out_bounds,
	Variable* input_data, Variable* data, const Shape& input_shape,
	const Shape& shape, const vector<Variable*>& variables, const string& name) {
	this->input_shape = input_shape;
	this->shape = shape;
	this->in_bounds = in_bounds;
	this->out_bounds = out_bounds;
	this->input_data = input_data;
	this->data = data;
	this->variables = variables;
	this->name = name;
}

Variable* Activation::operator()(Variable* inbound, bool is_training) {
	return this->activator->forward(is_training);
}

Variable* Activation::forward(bool is_training) {
	this->data = this->activator->forward(is_training);
	this->feed_variable_to_next_layer(this->data);
	return this->data;
}



ReLU::ReLU(bool inplace,
	const vector<Layer*>& in_bounds, const vector<Layer*>& out_bounds,
	Variable* input_data, Variable* data, const Shape& input_shape,
	const Shape& shape, const vector<Variable*>& variables, const string& name)
	: Activation(in_bounds, out_bounds, input_data, data, input_shape, shape,
		variables, name) {
	this->inplace = inplace;
}


Variable* ReLU::forward(bool is_training) {
	this->data = relu(this->input_data, this->inplace, is_training);
	this->feed_variable_to_next_layer(this->data);
	return this->data;
}

Variable* ReLU::operator()(Variable* inbound, bool is_training) {
	return relu(inbound, this->inplace, is_training);
}


Activation* get_activator(const string& name) {
	if (name == "relu") {
		return new ReLU();
	}
}