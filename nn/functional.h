#pragma once
#include "grad_fn.h"
#include<vector>
using namespace std;


// ######################## fully connected
// without bias
Variable* dense(Variable* inputs, Variable* weight, bool is_training = true,
	const string& name= "");
// with bias
Variable* dense(Variable* inputs, Variable* weight, Variable* bias, 
	bool is_training = true, const string& name = "");

//######################## convolutional
// without bias
//Variable* conv2d(Variable* inputs, Variable* weight,
//	const Shape& stride = Shape({ 1, 1 }), int padding = 1, const string& name = "");
//// with bias
//Variable* conv2d(Variable* inputs, Variable* weight, Variable* bias,
//	const Shape& stride = Shape({ 1, 1 }), int padding = 1, const string& name = "");

//######################## activator
// relu
Variable* relu(Variable* inputs, bool inplace = false, bool is_training = true,
	const string& name = "");
// sigmoid
Variable* sigmoid(Variable* inputs, bool is_training = true, const string& name = "");
// tanh
Variable* tanh(Variable* inputs, const string& name = "");
// softmax
Variable* softmax(Variable* inputs, const string& name = "");

void initialize_variables_grad(const initializer_list<Variable*>& l, bool is_training);