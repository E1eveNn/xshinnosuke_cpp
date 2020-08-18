#include "models.h"



Base::Base() {
	this->variables = unordered_set<Variable*>();
	this->trainable_variables = unordered_set<Variable*>();
}


void Base::train() {
	GlobalGraph::IS_TRAINING = true;
}

void Base::eval() {
	GlobalGraph::IS_TRAINING = false;
}

_Model::_Model() {
	this->loss = NULL;
	this->optimizer = NULL;
}

_Model::~_Model() {
	for (auto layer = this->graph.begin(); layer != this->graph.end();
		++layer) {
		if ((*layer) != NULL) {
			delete* layer;
			*layer = NULL;
		}
	}
}

Variable* _Model::operator()(Variable* x) {
	return this->forward(x);
}

void _Model::fit(Eigen::MatrixXf* x, Eigen::MatrixXf* y, int batch_size,
	int epochs, int verbose, bool shuffle) {
	Dataset train_set = Dataset(x, y);
	DataLoader train_loader = DataLoader(train_set, batch_size, shuffle);
	for (int epoch = 0; epoch < epochs; ++epoch) {
		for (auto it = train_loader.begin(); it != train_loader.end(); ++it) {
			this->train();
			Variable* inputs = (*it).first, * target = (*it).second;
			this->optimizer->zero_grad();
			Variable* out = this->forward(inputs);
			Variable* loss = this->loss->forward(out, target);
			this->backward(loss);
			this->optimizer->step();
			cout << "iter: " << epoch << " loss: " << loss->item() << " acc: " <<
				this->loss->calc_acc(out, target) << endl;
		}
	}
}

pair<float, float> _Model::evaluate(Eigen::MatrixXf* x, Eigen::MatrixXf* y,
	int batch_size) {
	Variable* y_true = new Variable(y);
	Variable* outputs = this->predict(x, batch_size);
	float acc = this->loss->calc_acc(outputs, y_true);
	float loss = this->loss->calc_loss(outputs, y_true);
	delete y_true;
	return make_pair(acc, loss);
}

Variable* _Model::predict(Eigen::MatrixXf* x, int batch_size) {
	this->eval();
	if (batch_size >= 0) {
		Eigen::MatrixXf* out = new Eigen::MatrixXf(x->rows(), x->cols());
		Variable* outputs = new Variable(out);
		Dataset test_set = Dataset(x, x);
		DataLoader test_loader = DataLoader(test_set, batch_size, false);
		int sp = 0;
		for (auto it = test_loader.begin(); it != test_loader.end(); ++it) {
			Variable* inputs = (*it).first;
			outputs->set_block(sp, 0, inputs->size(0), inputs->size(1),
				this->forward(inputs));
		}
		return outputs;
	}
	else {
		Variable* inputs = new Variable(x);
		Variable* outputs = this->forward(inputs);
		delete inputs;
		return outputs;
	}
}


void _Model::backward(Variable* loss) {
	loss->grad_fn(loss);
	for (auto layer = this->graph.rbegin();layer != this->graph.rend(); 
		++layer) {
		(*layer)->backward();
	}
}

Sequential::Sequential() {
	this->graph = vector<Layer*>();
}


Sequential::Sequential(const vector<Layer*>& graph) {
	this->graph = graph;
}

void Sequential::add(Layer* layer) {
	this->graph.push_back(layer);
}

void Sequential::compile(Optimizer* optimizer, Objective* loss,
	float lr) {
	if (this->graph.empty()) {
		throw "Graph is empty!";
	}
	unordered_map<string, int> name_dict;
	Layer* prev_layer = NULL;
	for (auto layer = this->graph.begin(); layer != this->graph.end(); ++layer) {
		(*layer)->connect(prev_layer);
		(*layer)->initial_params();
		if ((*layer)->name == "") {
			string className = (*layer)->get_className();
			for (int i = 0; i < className.size(); ++i) {
				className[i] = tolower(className[i]);
			}
			(*layer)->name = className +
				to_string(name_dict[(*layer)->get_className()]);
			name_dict[(*layer)->get_className()]++;
		}
		prev_layer = *layer;
		for (auto v = (*layer)->variables.begin(); 
			v != (*layer)->variables.end(); ++v) {
			this->variables.insert(*v);
		}
	}
	this->loss = loss;
	this->optimizer = optimizer;
}


void Sequential::compile(const string& optimizer, const string& loss,
	float lr) {
	if (this->graph.empty()) {
		throw "Graph is empty!";
	}
	unordered_map<string, int> name_dict;
	Layer* prev_layer = NULL;
	for (auto layer = this->graph.begin(); layer != this->graph.end(); ++layer) {
		(*layer)->connect(prev_layer);
		(*layer)->initial_params();
		if ((*layer)->name == "") {
			string className = (*layer)->get_className();
			for (int i = 0; i < className.size(); ++i) {
				className[i] = tolower(className[i]);
			}
			(*layer)->name = className +
				to_string(name_dict[(*layer)->get_className()]);
			name_dict[(*layer)->get_className()]++;
		}
		prev_layer = *layer;
		for (auto v = (*layer)->variables.begin();
			v != (*layer)->variables.end(); ++v) {
			this->variables.insert(*v);
		}
	}
	this->loss = get_objectives(loss);
	this->optimizer = get_optimizer(optimizer, this->variables, lr);
}


Variable* Sequential::forward(Variable* x) {
	this->graph[0]->input_data = x;
	for (auto layer = this->graph.begin(); layer != this->graph.end(); ++layer) {
		if ((*layer)->variables.empty()) {
			(*layer)->initial_params((*layer)->shape);
		}
		x = (*layer)->forward();
	}
	return x;
}


void Sequential::pop(int index) {
	cout << "successfully delete " << sliceString(
		typeid(this->graph.begin() + index).name(), 6) << "layer at position " 
		<<index << "th!" << endl;
	this->graph.erase(this->graph.begin() + index);
}


Module::Module() {

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
				auto params = (*n)->get_parameters();
				for (auto v = params.begin(); v != params.end(); ++v) {
					this->variables.insert(*v);
					if ((*v)->requires_grad)
						this->trainable_variables.insert(*v);
				}
				queue.push_back(*n);
				seen.insert(*n);
			}
		}

	}
}

Variable* Module::operator()(Variable* x) {
	Variable* outputs = this->forward(x);
	this->collect_variables(x);
	return outputs;
}

unordered_set<Variable*>* Module::parameters() {
	return &(this->variables);
}
