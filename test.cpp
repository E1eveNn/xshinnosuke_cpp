#include<iostream>
#include "shape.h"
#include "core.h"
#include "linear.h"
#include "models.h"
#include "functional.h"
#include<vector>

using namespace std;


class myNet : public Module {
public:
	myNet() {
		this->fc1 = new Linear(500);
		this->relu1 = new ReLU();
		this->fc2 = new Linear(100);
		this->relu2 = new ReLU();
		this->fc3 = new Linear(10);
	};
	Variable* forward(Variable* x, bool is_training) {
		Variable* out;
		out = (*(this->fc1))(x, is_training);
		out = (*(this->relu1))(out, is_training);
		out = (*(this->fc2))(out, is_training);
		out = (*(this->relu2))(out, is_training);
		out = (*(this->fc3))(out, is_training);
		return out;
	}
	
	Layer* fc1;
	Layer* relu1;
	Layer* fc2;
	Layer* relu2;
	Layer* fc3;
};


int main() {
	/*Shape s1({ 1 });
	Shape s2({ 3, 2, 5, 7 });
	
	cout << "s1: " << s1 << endl;
	cout << "s2: " << s2 << endl;
	cout << (s1 == s2);*/


	Eigen::MatrixXf x = Eigen::MatrixXf::Random(10, 784);
	Variable* inputs = new Variable(x);

	//Eigen::MatrixXf y = Eigen::MatrixXf::Random(4, 2);
	//Variable* weight = new Variable(y);

	//inputs->requires_grad = true;
	//weight->requires_grad = true;

	////// cout << n1;
	//Variable* out1 = dense(inputs, weight);
	//cout << out1 << endl;
	//Variable* out2 = relu(out1);
	//cout << out2 << endl;
	//out2->zero_grad();
	//out2->grad.fill(1);
	//cout << "1.out1's grad:" << out1->grad << endl;
	//out2->grad_fn(out2);
	//cout << "2.out1's grad:" << out1->grad << endl;
	//out1->grad_fn(out1);

	auto net = myNet();
	Variable* out = net(inputs);
	cout << out << endl;
}