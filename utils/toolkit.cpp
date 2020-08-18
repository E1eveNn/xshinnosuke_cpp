#include "toolkit.h"


Eigen::MatrixXf mat_log(Eigen::MatrixXf& mat) {
	int rows = mat.rows();
	int cols = mat.cols();
	Eigen::MatrixXf ret = Eigen::MatrixXf::Zero(rows, cols);
	for (int r = 0; r < rows; ++r) {
		for (int c = 0; c < cols; ++c) {
			ret(r, c) = log(mat(r, c));
		}
	}
	return ret;
}

char* sliceString(const char* origin, int sp, int ep) {
	if (ep == -1) {
		ep = strlen(origin);
	}
	if (ep > 0 && ep > sp) {
		int n = ep - sp;
		char* ret = new char[n + 1];
		for (int i = sp; i < ep; ++i)
			ret[i - sp] = origin[i];
		ret[n] = '\0';
		return ret;
	}

	return const_cast<char*>(origin);
}

//Eigen::MatrixXf* generate_matrix_pointer(int rows, int cols,
//	string type = "random") {
//	if (type == "random") {
//		
//	}
//}
