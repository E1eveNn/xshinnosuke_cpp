#include "objectives.h"


BinaryCrossEntropy::BinaryCrossEntropy() {

}


Variable* Objective::operator()(Variable* y_pred, Variable* y_true) {
	return this->forward(y_pred, y_true);
}

Variable* BinaryCrossEntropy::forward(Variable* y_pred, Variable* y_true) {
	y_pred->retain_grad();
	int r = y_true->data->rows();
	int c = y_true->data->cols();
	Eigen::MatrixXf ones = Eigen::MatrixXf::Ones(r, c);
	Eigen::MatrixXf onesMinusYpred = ones - *(y_pred->data);


	float loss_val = -((*(y_true->data)).array() * (mat_log(*(y_pred->data)).array()) + 
		(ones - *(y_true->data)).array() * mat_log(onesMinusYpred).array()).sum() / r;
	Eigen::MatrixXf* loss_mat = new Eigen::MatrixXf(1, 1);
	loss_mat->setConstant(loss_val);
	vector<Variable*> in_bounds{ y_pred, y_true };
	vector<Variable*> out_bounds;
	Variable* loss = new Variable(loss_mat, in_bounds, out_bounds, "", 
		y_pred->requires_grad | y_true->requires_grad);
	y_pred->out_bounds.push_back(loss);
	loss->grad_fn = BinaryCrossEntropyBackward;
	loss->set_grad_fn_name("BinaryCrossEntropyBackward");
	initialize_variables_grad({ y_pred, y_true }, GlobalGraph::IS_TRAINING);
	return loss;
}

float BinaryCrossEntropy::acc(Variable* y_pred, Variable* y_true) {
	int r = y_true->data->rows();
	int c = y_true->data->cols();
	float total_correct = 0;
	for (int i = 0; i < r; ++i) {
		for (int j = 0; j < c; ++j) {
			if ((*(y_pred->data))(i, j) < 0.5) {
				total_correct += ((*(y_true->data))(i, j) == 0);
			}
			else {
				total_correct += ((*(y_true->data))(i, j) == 1);
			}
		}
	}
	return total_correct / (r * c);
}