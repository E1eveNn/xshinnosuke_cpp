#include "grad_fn.h"


void AddBackward(Variable* outputs) {
	for (auto v = outputs->in_bounds.begin(); v != outputs->in_bounds.end();
		++v) {
		if ((*v)->requires_grad) {
			*((*v)->grad) += *(outputs->grad);
		}
	}
}


void DenseBackward(Variable* outputs) {
	Variable* inputs = outputs->in_bounds[0];
	Variable* weight = outputs->get_parameters()[0];
	if (inputs->requires_grad) {
		*(inputs->grad) += (*outputs->grad) * (weight->data->transpose());
	}
	if (weight->requires_grad) {
		*(weight->grad) += inputs->data->transpose() * (*(outputs->grad));
	}
}

void ReLUBackward(Variable* outputs) {
	Variable* inputs = outputs->in_bounds[0];
	if (inputs->requires_grad) {
		auto grad = *(outputs->grad);
		int rows = inputs->data->rows();
		int cols = inputs->data->cols();
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				if ((*inputs->data)(r, c) < 0) {
					grad(r, c) = 0;
				}
			}
		}
		*(inputs->grad) += grad;
	}
}

void SigmoidBackward(Variable* outputs) {
	Variable* inputs = outputs->in_bounds[0];
	if (inputs->requires_grad) {
		auto ones = *(outputs->data);
		ones.fill(1);
		*(inputs->grad) += Eigen::MatrixXf((*(outputs->grad)).array() * (*(outputs->data)).array()
			* (ones - (*outputs->data)).array());
	}
}


// objectives
void BinaryCrossEntropyBackward(Variable* outputs) {
	Variable* y_pred = outputs->in_bounds[0];
	Variable* y_true = outputs->in_bounds[1];
	int r = y_true->data->rows();
	int c = y_true->data->cols();
	// As Eigen doesn't support matrix / matrix, we manually apply matrix divison 
	// by for loop
	Eigen::MatrixXf grad = Eigen::MatrixXf::Zero(r, c);
	for (int i = 0; i < r; ++i) {
		for (int j = 0; j < c; ++j) {
			grad(i, j) = ((1 - (*(y_true->data))(i, j)) / 
				(1 - (*(y_pred->data))(i, j)) -
				(*(y_true->data))(i, j) / (*(y_pred->data))(i, j)) / r;
		}
	}
	if (y_true->requires_grad) {
		*(y_true->grad) += grad;
	}
	if (y_pred->requires_grad) {
		*(y_pred->grad) += grad;
	}
}