#pragma once
#include "core.h"


class Input :public Layer {
public:
	Input(const Shape& input_shape = Shape(), const vector<Layer*>& in_bounds = vector<Layer*>(),
		const vector<Layer*>& out_bounds = vector<Layer*>(),
		Variable* input_data = NULL, Variable* data = NULL,
		const Shape& shape = Shape(),
		const vector<Variable*>& variables = vector<Variable*>(),
		const string& name = "");
	virtual Variable* forward(bool is_training = true);
};