#include "optimizers.h"


void Optimizer::zero_grad() {
	for (unordered_set<Variable*>::iterator it = this->variables->begin();
		it != this->variables->end(); ++it) {
		(*it)->zero_grad();
	}
}


SGD::SGD(unordered_set<Variable*>* variables, float lr) {
	this->variables = variables;
	this->lr = lr;
	this->iterations = 0;
}

SGD::SGD() {
	this->iterations = 0;
}


void SGD::step() {
	for (unordered_set<Variable*>::iterator it = this->variables->begin();
		it != this->variables->end(); ++it) {
		if ((*it)->requires_grad) {
			/*Eigen::MatrixXf lr_mat = 
				Eigen::MatrixXf::Constant((*it)->data->rows(), 
					(*it)->data->cols(), this->lr);*/
			(*(*it)->data) = (*(*it)->data) - this->lr * (*(*it)->grad);
		}
	}
	this->iterations++;
}


Optimizer* get_optimizer(const string& name,
	unordered_set<Variable*>& variables) {
	if (name == "sgd" || name == "SGD") {
		return new SGD(&variables);
	}
}

Optimizer* get_optimizer(const string& name,
	unordered_set<Variable*>& variables, float lr) {
	if (name == "sgd" || name == "SGD") {
		return new SGD(&variables, lr);
	}
}