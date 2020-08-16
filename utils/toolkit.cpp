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

//Eigen::MatrixXf* generate_matrix_pointer(int rows, int cols,
//	string type = "random") {
//	if (type == "random") {
//		
//	}
//}
