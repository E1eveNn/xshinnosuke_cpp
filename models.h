#pragma once
#include<iomanip>
#include<unordered_set>
#include "nn/optimizers.h"
#include "nn/objectives.h"
#include "utils/data.h"
using namespace std;

class Base {
public:
	Base();
	virtual Variable* forward(Variable* x) = 0;
	void train();
	void eval();
protected:
	unordered_set<Variable*> variables;
	unordered_set<Variable*> trainable_variables;
};


class Module : public Base {
public:
	Module();
	Variable* operator()(Variable* x);
	unordered_set<Variable*>* parameters();


protected:
	void collect_variables(Variable* x);
};


class _Model :public Base {
public:
	_Model();
	~_Model();
	Optimizer* optimizer;
	Objective* loss;
	vector<Layer*> graph;

	virtual void compile(Optimizer* optimizer, Objective* loss, 
		float lr = 0.1) = 0;
	virtual void compile(const string& optimizer, const string& loss,
		float lr = 0.1) = 0;
	virtual Variable* forward(Variable* x) = 0;

	void fit(Eigen::MatrixXf* x, Eigen::MatrixXf* y, int batch_size = 0,
		int epochs = 1, int verbose = 1, bool shuffle = true);
	Variable* operator()(Variable* x);
	void backward(Variable* loss);
	pair<float, float> evaluate(Eigen::MatrixXf* x, Eigen::MatrixXf* y, 
		int batch_size = 0);
	Variable* predict(Eigen::MatrixXf* x, int batch_size = 0);

};

class Sequential :public _Model {
	// overload cout
	friend ostream& operator<<(ostream& os, Sequential& n) {
		int bar_nums = 75;
		for (int i = 0; i < bar_nums; ++i) {
			cout << "*";
		}
		cout << endl;
		/*cout << "Layer(type)\t\t" << "Output Shape\t" << "Param\t" <<
			"Connected to" << endl;*/
		cout << std::left << setw(25) << "Layer(type)" << std::left
			<< setw(20) << "Output Shape" << std::left << setw(10) << "Param"
			<< std::left << setw(15) << "Connected to" << endl;

		for (int i = 0; i < bar_nums; ++i) {
			cout << "#";
		}
		cout << endl;
		if (n.graph.empty()) {
			cout << "Please compile Model!" << endl;
			return os;
		}
		for (auto layer = n.graph.begin(); layer != n.graph.end(); ++layer) {
			string layer_name = (*layer)->name + " (" + 
				(*layer)->get_className() + ")";
			long paramas = (*layer)->params_count();
			bool first = true;
			if ((*layer)->in_bounds.size() != 0) {
				for (auto prev_layer = (*layer)->in_bounds.begin();
					prev_layer != (*layer)->in_bounds.end(); ++prev_layer) {
					if (first) {
						cout << std::left << setw(25) << layer_name <<
							std::left << setw(20) << "(None, " + 
							to_string((*layer)->shape[0]) + ")" <<
							std::left << setw(10) << paramas << std::left
							<< setw(15) << (*prev_layer)->name << endl;
						first = false;
					}
					else {
						cout << std::left << setw(25) << std::left << 
							setw(20) << std::left << setw(10) << std::left 
							<< setw(15) << (*prev_layer)->name << endl;
					}
				}
			}
			else {
				cout << std::left << setw(25) << layer_name << std::left
					<< setw(20) << "(None, " + to_string((*layer)->shape[0]) 
					+ ")" << std::left << setw(10) << paramas  << endl;
			}
			for (int i = 0; i < bar_nums; ++i) {
				cout << "-";
			}
			cout << endl;
		}
		for (int i = 0; i < bar_nums; ++i) {
			cout << "*";
		}
		cout << endl;
		long total_params = 0;
		long trainable_params = 0;
		for (auto v = n.variables.begin(); v != n.variables.end(); ++v) {
			total_params += (*v)->data->size();
			if ((*v)->requires_grad) {
				trainable_params += (*v)->data->size();
			}
		}
		cout << "Total params: " << total_params << endl;
		cout << "Trainable params: " << trainable_params << endl;
		cout << "Non-trainable params: " << total_params - trainable_params
			<< endl;
		return os;
	}
public:
	Sequential();
	Sequential(const vector<Layer*>& graph);
	void add(Layer* layer);
	virtual void compile(Optimizer* optimizer, Objective* loss,
		float lr = 0.1);
	virtual void compile(const string& optimizer, const string& loss,
		float lr = 0.1);
	virtual Variable* forward(Variable* x);
	void pop(int index);
};