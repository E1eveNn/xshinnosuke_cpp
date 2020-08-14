#include "base.h"


Input::Input(const Shape& input_shape, const vector<Layer*>& in_bounds,
	const vector<Layer*>& out_bounds, Variable* input_data, Variable* data,
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

Variable* Input::forward(bool is_training) {
	this->data = this->input_data;
	this->feed_variable_to_next_layer(this->data);
	return this->data;
}