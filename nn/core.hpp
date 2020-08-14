#pragma once
#include <vector>
#include<Eigen/Dense>
#include<iostream>
#include<string>
#include<unordered_map>
#include "shape.h"
using namespace std;

typedef Eigen::Matrix<int, 1, 1> Vector1I;
typedef Eigen::Matrix<float, 1, 1> Vector1f;
typedef Eigen::Matrix<double, 1, 1> Vector1d;

// Tensor Class
class Variable {
	// overload cout
	friend ostream& operator<<(ostream& os, Variable* n) {
		string requires_grad = n->requires_grad == 0 ? "false" : "true";
		cout << n->sliceString(typeid(n).name(), 6) << '('
			<< n->data << ", requires_grad=" <<
			requires_grad << ", grad_fn=<" << n->grad_fn_name << ">)" << endl;
		return os;
	}

public:
	Variable(Eigen::MatrixXf& data, const vector<Variable*>& in_bounds = vector<Variable*>(),
		const vector<Variable*>& out_bounds = vector<Variable*>(),
		const string& name = "", bool requires_grad = false, 
		const string& dtype = "float32");
	/*Variable(Eigen::VectorXf& data, const vector<Variable*>& in_bounds = vector<Variable*>(),
		const vector<Variable*>& out_bounds = vector<Variable*>(),
		const string& name = "", bool requires_grad = false,
		const string& dtype = "float32");*/
	/*Variable(int& data, const vector<Variable&>& in_bounds = vector<Variable&>(),
		const vector<Variable&>& out_bounds = vector<Variable&>(), const string& name = "", bool requires_grad = false,
		const string& dtype = "float32");
	Variable(float& data, const vector<Variable&>& in_bounds = vector<Variable&>(),
		const vector<Variable&>& out_bounds = vector<Variable&>(), const string& name = "", bool requires_grad = false,
		const string& dtype = "float32");
	Variable(double& data, const vector<Variable&>& in_bounds = vector<Variable&>(),
		const vector<Variable&>& out_bounds = vector<Variable&>(), const string& name = "", bool requires_grad = false,
		const string& dtype = "float32")*/;

	// storage tensor
	Eigen::MatrixXf data;
	// tensor's shape
	Shape shape;
	// tensor'name
	string name;
	// record whether requires gradadient
	bool requires_grad;
	// tensor's gradient
	Eigen::MatrixXf grad;

	vector<Variable*> in_bounds;
	vector<Variable*> out_bounds;

	// grad_fn
	void (*grad_fn)(Variable *);

	// function
	void retain_grad();

	void backward(Eigen::MatrixXf& gradients);
	void backward();

	int size();
	void zero_grad();
	void set_grad_fn_name(const string& name);
	// overload operator
	Variable& operator + (const Variable& other);

protected:
	unordered_map<string, float> cache;
	bool retain;
	string grad_fn_name;
	char* sliceString(const char* origin, int sp, int ep = -1);
};



class Constant : public Variable {
public:
	Constant(Eigen::MatrixXf& data, const vector<Variable*>* in_bounds = NULL,
		const vector<Variable*>* out_bounds = NULL, const string& name = "",
		bool requires_grad = false, const string& dtype = "float32");
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

	virtual void initial_params();
	virtual Shape compute_output_shape(Shape& input_shape);
	int params_count();
	virtual Variable* forward(bool is_training=true) = 0;
	virtual void backward();
	//virtual Layer* operator()(Layer* inbound);
	virtual Variable* operator()(Variable* inbound, bool is_training = true) = 0;
protected:
	void feed_variable_to_next_layer(Variable* data);
};