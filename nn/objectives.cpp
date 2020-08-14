#include "objectives.h"

Variable* Objective::operator()(Variable* y_pred, Variable* y_true) {
	return this->forward(y_pred, y_true);
}

Variable* BinaryCrossEntropy::forward(Variable* y_pred, Variable* y_true) {
	y_pred->retain_grad();
	int r = y_true->data.rows();
	int c = y_true->data.cols();
	Eigen::MatrixXf::Ones ones(r, c);
	Eigen::MatrixXf onesMinusYpred = ones - y_pred->data;
	float loss_val = (-y_true->data * mat_log(y_pred->data) -
		(ones - y_true->data * mat_log(onesMinusYpred))).sum() / r;
	Eigen::MatrixXf loss_mat = Eigen::MatrixXf::Constant(loss_val);
	Variable* loss = new Variable(loss_mat, vector<Variable*>(y_pred, y_true));
	y_pred->out_bounds.push_back(loss);
	loss->grad_fn = BinaryCrossEntropyBackward;
	loss->set_grad_fn_name("BinaryCrossEntropyBackward");
	return loss;
}

float BinaryCrossEntropy::acc(Variable* y_pred, Variable* y_true) {
	int r = y_true->data.rows();
	int c = y_true->data.cols();
	long total_correct = 0;
	for (int i = 0; i < r; ++r) {
		for (int j = 0; j < c; ++c) {
			if (y_pred->data(i, j) < 0.5) {
				total_correct += (y_true->data(i, j) == 0);
			}
			else {
				total_correct += (y_true->data(i, j) == 1);
			}
		}
	}
	return total_correct / (r * c);
}