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

Variable* Activation::operator()(Variable* inbound) {
	return this->activator->forward();
}

Variable* Activation::forward() {
	this->data = this->activator->forward();
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
	this->className = "ReLU";
}


Variable* ReLU::forward() {
	this->data = relu(this->input_data, this->inplace, GlobalGraph::IS_TRAINING);
	this->feed_variable_to_next_layer(this->data);
	return this->data;
}

Variable* ReLU::operator()(Variable* inbound) {
	return relu(inbound, this->inplace, GlobalGraph::IS_TRAINING);
}


Sigmoid::Sigmoid(const vector<Layer*>& in_bounds, const vector<Layer*>& out_bounds,
	Variable* input_data, Variable* data, const Shape& input_shape,
	const Shape& shape, const vector<Variable*>& variables, const string& name)
	: Activation(in_bounds, out_bounds, input_data, data, input_shape, shape,
		variables, name) {
	this->className = "Sigmoid";
}


Variable* Sigmoid::forward() {
	this->data = sigmoid(this->input_data, GlobalGraph::IS_TRAINING);
	this->feed_variable_to_next_layer(this->data);
	return this->data;
}

Variable* Sigmoid::operator()(Variable* inbound) {
	return sigmoid(inbound, GlobalGraph::IS_TRAINING);
}


Activation* get_activator(const string& name) {
	if (name == "relu") {
		return new ReLU();
	}
	else if(name == "sigmoid") {
		return new Sigmoid();
	}
}