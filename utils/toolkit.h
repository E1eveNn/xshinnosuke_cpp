#pragma once

#include<Eigen/Dense>
#include<unordered_set>
#include<string>
#include <cstring>
#include<cmath>
using namespace std;


Eigen::MatrixXf mat_log(Eigen::MatrixXf& mat);
Eigen::MatrixXf* generate_matrix_pointer(int rows, int cols, 
	string type = "random");

