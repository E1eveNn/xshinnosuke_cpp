#include "optimizers.h"


void Optimizer::zero_grad() {
	for (vector<Variable>::iterator it = this->variables.begin(); 
		it != this->variables.end(); ++it) {
		it->zero_grad();
	}
}


SGD::SGD(vector<Variable>& variables, float lr) {
	this->variables = variables;
	this->lr = lr;
	this->iterations = 0;
}


void SGD::step() {
	for (vector<Variable>::iterator it = this->variables.begin();
		it != this->variables.end(); ++it) {
		if (it->requires_grad) {
			it->data = it->data - this->lr * it->grad;
		}
	}
	this->iterations++;
}