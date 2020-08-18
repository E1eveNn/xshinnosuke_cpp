#include "functional.h"

// math operations
Variable* Variable::operator + (Variable* other) {
	if (GlobalGraph::INPUTS == NULL) {
		GlobalGraph::INPUTS = this;
	}
	return badd({ this, other }, GlobalGraph::IS_TRAINING);
}


Variable* badd(const initializer_list<Variable*>& l, bool is_training, 
	const string& name) {
	int r = (*(l.begin()))->data->rows();
	int c = (*(l.begin()))->data->cols();
	Eigen::MatrixXf* data = new Eigen::MatrixXf(r, c);
	data->setZero();
	vector<Variable*> in_bounds;
	bool requires_grad = false;
	for (auto v = l.begin(); v != l.end(); ++v) {
		if ((*v)->data->rows() != r || (*v)->data->cols() != c) {
			throw "add() requires all the variables hold the same shape!";
		}
		*data += *((*v)->data);
		in_bounds.push_back(*v);
		requires_grad = requires_grad | (*v)->requires_grad;
	}
	vector<Variable*> out_bounds;
	Variable* outputs = new Variable(data, in_bounds, out_bounds, name,
		requires_grad);
	for (auto v = l.begin(); v != l.end(); ++v) {
		(*v)->out_bounds.push_back(outputs);
		if (is_training && outputs->requires_grad) {
			outputs->grad_fn = AddBackward;
			outputs->set_grad_fn_name("AddBackward");
		}
	}
	initialize_variables_grad(l, is_training);
	return outputs;
}



Variable* dense(Variable* inputs, Variable* weight, bool is_training, 
	const string& name) {
	if (GlobalGraph::INPUTS == NULL) {
		GlobalGraph::INPUTS = inputs;
	}
	Eigen::MatrixXf out = (*inputs->data) * (*weight->data);
	Eigen::MatrixXf* data = new Eigen::MatrixXf(out);
	vector<Variable*> in_bounds = {inputs};
	vector<Variable*> out_bounds;
	Variable* outputs = new Variable(data, in_bounds, out_bounds, name,
		inputs->requires_grad | weight->requires_grad);
	inputs->out_bounds.push_back(outputs);
	outputs->set_parameters({ weight });
	if (is_training && outputs->requires_grad) {
		outputs->grad_fn = DenseBackward;
		outputs->set_grad_fn_name("DenseBackward");
		initialize_variables_grad({ inputs, weight }, is_training);
	}
	return outputs;
}


Variable* dense(Variable* inputs, Variable* weight, Variable* bias, 
	bool is_training, const string& name) {
	if (GlobalGraph::INPUTS == NULL) {
		GlobalGraph::INPUTS = inputs;
	}
	Eigen::MatrixXf out = (*inputs->data) * (*weight->data);
	//out.rowwise() += bias->data->array();
	for (int r = 0; r < out.rows(); ++r)
		out.row(r) += *(bias->data);
	Eigen::MatrixXf* data = new Eigen::MatrixXf(out);
	vector<Variable*> in_bounds = { inputs };
	vector<Variable*> out_bounds;
	Variable* outputs = new Variable(data, in_bounds, out_bounds, name,
		inputs->requires_grad | weight->requires_grad | bias->requires_grad);
	inputs->out_bounds.push_back(outputs);
	outputs->set_parameters({ weight, bias });
	if (is_training && outputs->requires_grad) {
		outputs->grad_fn = DenseBackward;
		outputs->set_grad_fn_name("DenseBackward");
		initialize_variables_grad({ inputs, weight, bias }, is_training);
	}
	return outputs;
}


Variable* relu(Variable* inputs, bool inplace, bool is_training, 
	const string& name) {
	if (GlobalGraph::INPUTS == NULL) {
		GlobalGraph::INPUTS = inputs;
	}
	if (inplace) {
		*(inputs->data) = inputs->data->cwiseMax(0);
		return inputs;
	}

	vector<Variable*> in_bounds = { inputs };
	vector<Variable*> out_bounds;
	Eigen::MatrixXf out = inputs->data->cwiseMax(0);
	Eigen::MatrixXf* data = new Eigen::MatrixXf(out);
	Variable* outputs = new Variable(data, in_bounds, out_bounds, name,
		inputs->requires_grad);
	inputs->out_bounds.push_back(outputs);
	if (is_training && outputs->requires_grad) {
		outputs->grad_fn = ReLUBackward;
		outputs->set_grad_fn_name("ReLUBackward");
		initialize_variables_grad({ inputs }, is_training);
	}
	return outputs;
}


Variable* sigmoid(Variable* inputs, bool is_training, const string& name) {
	if (GlobalGraph::INPUTS == NULL) {
		GlobalGraph::INPUTS = inputs;
	}
	Eigen::MatrixXf out = 1. / (1 + ((-inputs->data->array()).exp()));
	Eigen::MatrixXf* data = new Eigen::MatrixXf(out);
	vector<Variable*> in_bounds = { inputs };
	vector<Variable*> out_bounds;
	Variable* outputs = new Variable(data, in_bounds, out_bounds, name,
		inputs->requires_grad);
	inputs->out_bounds.push_back(outputs);
	if (outputs->requires_grad) {
		outputs->grad_fn = SigmoidBackward;
		outputs->set_grad_fn_name("SigmoidBackward");
		initialize_variables_grad({ inputs }, is_training);
	}
	return outputs;
}

//Variable* tanh(Variable* inputs, const string& name = "") {
//	Eigen::MatrixXf out = 1 - (-2 * inputs->data).array().exp()
//		/ (1 + (-2 * inputs->data.array().exp()));
//	vector<Variable*> in_bounds = { inputs };
//	vector<Variable*> out_bounds;
//	return new Variable(out, in_bounds, out_bounds, name,
//		inputs->requires_grad);
//}

//Variable* softmax(Variable* inputs, const string& name) {
//	// more stable softmax
//	Eigen::MatrixXf shiftx = inputs->data - Vector1f(inputs->data.maxCoeff());
//	Eigen::MatrixXf out = shiftx.exp() / (shiftx.exp())
//}

//Variable* conv2d(Variable* inputs, Variable* weight,
//	const Shape& stride, int padding, const string& name) {
//	Shape input_shape = inputs->shape;
//}

void initialize_variables_grad(const initializer_list<Variable*>& l, bool is_training) {
	if (is_training) {
		for (auto v = l.begin(); v != l.end(); ++v) {
			if ((*v)->requires_grad) {
				(*v)->zero_grad();
			}
		}
	}
}