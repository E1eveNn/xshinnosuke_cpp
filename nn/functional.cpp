#include "functional.h"

Variable* dense(Variable* inputs, Variable* weight, bool is_training, 
	const string& name) {
	Eigen::MatrixXf out = inputs->data * weight->data;

	vector<Variable*> in_bounds = {inputs, weight};
	vector<Variable*> out_bounds;
	Variable* outputs = new Variable(out, in_bounds, out_bounds, name,
		inputs->requires_grad | weight->requires_grad);
	inputs->out_bounds.push_back(outputs);
	if (is_training && outputs->requires_grad) {
		outputs->grad_fn = DenseBackward;
		outputs->set_grad_fn_name("DenseBackward");
		initialize_variables_grad({ inputs, weight }, is_training);
	}
	return outputs;
}


Variable* dense(Variable* inputs, Variable* weight, Variable* bias, 
	bool is_training, const string& name) {

	Eigen::MatrixXf out = (inputs->data * weight->data);
	for (int r = 0; r < out.rows(); ++r)
		out.row(r) += bias->data;
	vector<Variable*> in_bounds = { inputs, weight, bias };
	vector<Variable*> out_bounds;
	Variable* outputs = new Variable(out, in_bounds, out_bounds, name,
		inputs->requires_grad | weight->requires_grad | bias->requires_grad);
	inputs->out_bounds.push_back(outputs);
	if (is_training && outputs->requires_grad) {
		outputs->grad_fn = DenseBackward;
		outputs->set_grad_fn_name("DenseBackward");
		initialize_variables_grad({ inputs, weight, bias }, is_training);
	}
	return outputs;
}


Variable* relu(Variable* inputs, bool inplace, bool is_training, const string& name) {
	if (inplace) {
		inputs->data = inputs->data.cwiseMax(0);
		return inputs;
	}

	vector<Variable*> in_bounds = { inputs };
	vector<Variable*> out_bounds;
	Eigen::MatrixXf out = inputs->data.cwiseMax(0);
	Variable* outputs = new Variable(out, in_bounds, out_bounds, name,
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
	Eigen::MatrixXf out = 1. / (1 + (-inputs->data.array().exp()));
	vector<Variable*> in_bounds = { inputs };
	vector<Variable*> out_bounds;
	Variable* outputs = new Variable(out, in_bounds, out_bounds, name,
		inputs->requires_grad);
	inputs->out_bounds.push_back(outputs);
	if (outputs->requires_grad) {
		outputs->grad_fn = ReLUBackward;
		outputs->set_grad_fn_name("ReLUBackward");
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