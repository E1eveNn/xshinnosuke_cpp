#include "initializers.h"


Variable* Zeros::operator()(int rows, int cols) {
	Eigen::MatrixXf out = Eigen::MatrixXf::Zero(rows, cols);
	return new Variable(out);
}

Variable* Zeros::operator()(const Shape& shape) {
	/*int rows = shape[0];
	int cols = shape[1];*/
	int rows = shape.data[0];
	int cols = shape.data[1];
	Eigen::MatrixXf out = Eigen::MatrixXf::Zero(rows, cols);
	return new Variable(out);
}

//Variable* Zeros::operator()(int length) {
//	Eigen::VectorXf out = Eigen::VectorXf::Zero(length);
//	return new Variable(out);
//}


Variable* Ones::operator()(int rows, int cols) {
	Eigen::MatrixXf out = Eigen::MatrixXf::Ones(rows, cols);
	return new Variable(out);
}

Variable* Ones::operator()(const Shape& shape) {
	int rows = shape.data[0];
	int cols = shape.data[1];
	Eigen::MatrixXf out = Eigen::MatrixXf::Ones(rows, cols);
	return new Variable(out);
}

Variable* Random::operator()(int rows, int cols) {
	Eigen::MatrixXf out = Eigen::MatrixXf::Random(rows, cols);
	return new Variable(out);
}

Variable* Random::operator()(const Shape& shape) {
	int rows = shape.data[0];
	int cols = shape.data[1];
	Eigen::MatrixXf out = Eigen::MatrixXf::Random(rows, cols);
	return new Variable(out);
}


Initializer* get_initializer(const string& name) {
	if (name == "zeros") {
		return new Zeros();
	}
	else if (name == "ones") {
		return new Ones();
	}
	else if (name == "random") {
		return new Random();
	}
}