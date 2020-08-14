#include "models.h"
#include<unordered_set>
using namespace std;


Base::Base() {
	this->is_training = true;
}


void Base::train() {
	this->is_training = true;
}

void Base::eval() {
	this->is_training = false;
}


Module::Module() {
	this->variables = vector<Variable*> ();
	this->trainable_variables = vector<Variable*> ();
}

void Module::collect_variables(Variable* x) {
	vector<Variable*> queue = { x };
	unordered_set<Variable*> seen;
	seen.insert(x);
	while (!queue.empty()) {
		Variable* vertex = queue.back();
		queue.pop_back();
		for (auto n = vertex->out_bounds.begin(); n != vertex->out_bounds.end(); ++n) {
			if (seen.count(*n) == 0) {
				for (auto v = (*n)->in_bounds.begin(); v != (*n)->in_bounds.end(); ++v) {
					this->variables.push_back(*v);
					if ((*v)->requires_grad)
						this->trainable_variables.push_back(*v);
				}
				queue.push_back(*n);
				seen.insert(*n);
			}
		}

	}
}

Variable* Module::operator()(Variable* x) {
	Variable* outputs = this->forward(x, this->is_training);
	this->collect_variables(x);
	return outputs;
}

vector<Variable*>& Module::parameters() {
	return this->variables;
}
