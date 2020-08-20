#include "initializers.h"


Variable* Zeros::operator()(int rows, int cols) {
	Eigen::MatrixXf* out = new Eigen::MatrixXf(rows, cols);
	out->setZero();
	return new Variable(out);
}

Variable* Zeros::operator()(const Shape& shape) {
	/*int rows = shape[0];
	int cols = shape[1];*/
	int rows = shape.data[0];
	int cols = shape.data[1];
	Eigen::MatrixXf* out = new Eigen::MatrixXf(rows, cols);
	out->setZero();
	return new Variable(out);
}

//Variable* Zeros::operator()(int length) {
//	Eigen::VectorXf out = Eigen::VectorXf::Zero(length);
//	return new Variable(out);
//}


Variable* Ones::operator()(int rows, int cols) {
	Eigen::MatrixXf* out = new Eigen::MatrixXf(rows, cols);
	out->setOnes();
	return new Variable(out);
}

Variable* Ones::operator()(const Shape& shape) {
	int rows = shape.data[0];
	int cols = shape.data[1];
	Eigen::MatrixXf* out = new Eigen::MatrixXf(rows, cols);
	out->setOnes();
	return new Variable(out);
}

Variable* Random::operator()(int rows, int cols) {
	Eigen::MatrixXf* out = new Eigen::MatrixXf(rows, cols);
	out->setRandom();
	return new Variable(out);
}

Variable* Random::operator()(const Shape& shape) {
	int rows = shape.data[0];
	int cols = shape.data[1];
	Eigen::MatrixXf* out = new Eigen::MatrixXf(rows, cols);
	out->setRandom();
	return new Variable(out);
}

//Variable* Binominal::operator()(int rows, int cols) {
//	Eigen::MatrixXf* out = new Eigen::MatrixXf(rows, cols);
//	out->setZero();
//	for (int i = 0; i < rows; ++i) {
//		for (int j = 0; j < cols; ++j) {
//			float n = (rand() % 10) / 10;
//			if (n >= 0.5) {
//				(*out)(i, j) = 1;
//			}
//		}
//	}
//	return new Variable(out);
//}
//
//Variable* Binominal::operator()(const Shape& shape) {
//	int rows = shape.data[0];
//	int cols = shape.data[1];
//	Eigen::MatrixXf* out = new Eigen::MatrixXf(rows, cols);
//	out->setZero();
//	for (int i = 0; i < rows; ++i) {
//		for (int j = 0; j < cols; ++j) {
//			float n = (rand() % 10) / 10;
//			if (n >= 0.5) {
//				(*out)(i, j) = 1;
//			}
//		}
//	}
//	return new Variable(out);
//}


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