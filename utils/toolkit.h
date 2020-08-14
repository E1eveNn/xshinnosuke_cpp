#pragma once
#include "core.h"
#include "activators.h"
#include<unordered_set>
#include <cstring>
#include<cmath>


void initialize_variables_grad(const initializer_list<Variable*>& l, bool is_training);
Eigen::MatrixXf& mat_log(Eigen::MatrixXf& mat);

template <class T>
vector<Variable*> topological_sort(T* inputs, T* outputs);