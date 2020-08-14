#include "grad_fn.h"

void DenseBackward(Variable* outputs) {
	Variable* inputs = outputs->in_bounds[0];
	Variable* weight = outputs->in_bounds[1];
	if (inputs->requires_grad) {
		inputs->grad += outputs->grad * (weight->data.transpose());
	}
	if (weight->requires_grad) {
		weight->grad += inputs->data.transpose() * outputs->grad;
	}
}

void ReLUBackward(Variable* outputs) {
	Variable* inputs = outputs->in_bounds[0];
	if (inputs->requires_grad) {
		auto grad = outputs->grad;
		int rows = inputs->data.rows();
		int cols = inputs->data.cols();
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				if (inputs->data(r, c) < 0) {
					grad(r, c) = 0;
				}
			}
		}
		inputs->grad += grad;
	}
}

void SigmoidBackward(Variable* outputs) {
	Variable* inputs = outputs->in_bounds[0];
	if (inputs->requires_grad) {
		auto ones = outputs->data;
		ones.fill(1);
		inputs->grad += outputs->grad * outputs->data * (ones - outputs->data);
	}
}