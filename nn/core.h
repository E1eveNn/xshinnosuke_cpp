#pragma once
#include <vector>
#include<Eigen/Dense>
#include<iostream>
#include<string>
#include<unordered_map>
#include<unordered_set>
#include<algorithm>
#include "../utils/shape.h"
#include "../utils/toolkit.h"
using namespace std;

#ifndef BlockType
// #define BlockType Eigen::Block<Eigen::Matrix<float, -1, -1, 0, -1, -1>, -1, -1, 0>;
typedef Eigen::Block<Eigen::Matrix<float, -1, -1, 0, -1, -1>, -1, -1, 0> BlockType;
#endif

#ifndef MatrixType
//#define MatrixType Eigen::MatrixXf;
typedef Eigen::MatrixXf MatrixType;
#endif

//typedef Eigen::Block<Eigen::Matrix<float, -1, -1, 0, -1, -1>, -1, -1, 0> BlockType;
//typedef Eigen::MatrixXf MatrixType;
typedef Eigen::Matrix<int, 1, 1> Vector1I;
typedef Eigen::Matrix<float, 1, 1> Vector1f;
typedef Eigen::Matrix<double, 1, 1> Vector1d;




// Tensor Class
class Variable {
	// overload cout
	friend ostream& operator<<(ostream& os, Variable* n) {
		string requires_grad = n->requires_grad == 0 ? "false" : "true";
		cout << sliceString(typeid(n).name(), 6) << '('
			<< *(n->data) << ", requires_grad=" <<
			requires_grad << ", grad_fn=<" << n->grad_fn_name << ">)";
		return os;
	}

public:
	Variable(MatrixType* data, const vector<Variable*>& in_bounds = vector<Variable*>(),
		const vector<Variable*>& out_bounds = vector<Variable*>(),
		const string& name = "", bool requires_grad = false);
	/*Variable(BlockType* data, const vector<Variable*>& in_bounds = vector<Variable*>(),
		const vector<Variable*>& out_bounds = vector<Variable*>(),
		const string& name = "", bool requires_grad = false);*/

	Variable(const Variable& v);
	~Variable();

	// storage tensor
	Eigen::MatrixXf* data;
	// tensor's shape
	Shape shape;
	// tensor'name
	string name;
	// record whether requires gradadient
	bool requires_grad;
	// tensor's gradient
	Eigen::MatrixXf* grad;

	vector<Variable*> in_bounds;
	vector<Variable*> out_bounds;

	vector<Variable*>& get_parameters();
	void set_parameters(const initializer_list<Variable*>& l);

	// grad_fn
	void (*grad_fn)(Variable *);

	// function
	void retain_grad();

	void reset();

	void requires_grad_(bool requires_grad);

	Eigen::MatrixXf& item();

	void backward(Eigen::MatrixXf* gradients);
	void backward();

	Shape& size();
	int size(int index);
	void zero_grad();
	void set_grad_fn_name(const string& name);

	void set_block(int i, int j, int h, int w, Variable* block);
	// overload operator
	// void operator delete(void* p);
	Variable* operator + (Variable* other);
	bool retain;
	bool data_delete_flag;
protected:
	unordered_map<string, float> cache;
	string grad_fn_name;
	vector<Variable*> parameters;
};



class Constant : public Variable {
public:
	Constant(Eigen::MatrixXf& data, const vector<Variable*>* in_bounds = NULL,
		const vector<Variable*>* out_bounds = NULL, const string& name = "",
		bool requires_grad = false);
	const Eigen::MatrixXf data;
};


class Layer {
public:
	// friend ostream& operator<<(ostream& os, const Layer& l);
	Variable* input_data;
	Variable* data;
	string name;
	Shape input_shape;
	Shape shape;
	vector<Layer*> in_bounds;
	vector<Layer*> out_bounds;
	vector<Variable*> variables;

	~Layer();
	virtual void initial_params(Shape& input_shape);
	virtual void initial_params();
	virtual Shape compute_output_shape(Shape& input_shape);
	virtual Variable* forward() = 0;
	virtual void backward();
	virtual Layer* operator()(Layer* inbound);
	virtual Variable* operator()(Variable* inbound) = 0;

	int params_count();
	void connect(Layer* inbound = NULL);

	string& get_className();
protected:
	void feed_variable_to_next_layer(Variable* data);
	string className;
};

namespace GlobalGraph {
	// global parameters
	extern Variable* INPUTS;
	extern Variable* OUTPUTS;
	extern vector<Variable*> GRAPH;
	extern bool IS_TRAINING;

	template <class T>
	vector<T*> topological_sort(T* inputs, T* outputs);
	void reset_graph();
};


