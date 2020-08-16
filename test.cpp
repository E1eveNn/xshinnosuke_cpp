#include<iostream>
#include "layers/linear.h"
#include "models.h"
#include "nn/functional.h"
#include "nn/objectives.h"
#include "nn/optimizers.h"
#include<vector>

using namespace std;


class myNet : public Module {
public:
	myNet(int n_classes) {
		this->fc1 = new Linear(10);
		this->relu1 = new ReLU();
		this->fc2 = new Linear(5);
		this->relu2 = new ReLU();
		this->fc3 = new Linear(n_classes, "sigmoid");

	};
	Variable* forward(Variable* x) {
		Variable* out;
		out = (*(this->fc1))(x);
		out = (*(this->relu1))(out);
		out = (*(this->fc2))(out);
		out = (*(this->relu2))(out);
		out = (*(this->fc3))(out);
		out = sigmoid(out);
		return out;
	}
	
	Layer* fc1;
	Layer* relu1;
	Layer* fc2;
	Layer* relu2;
	Layer* fc3;

};




int main() {


	Eigen::MatrixXf x = Eigen::MatrixXf::Random(10, 10);
	Variable* inputs = new Variable(&x);
	inputs->requires_grad_(true);
	Eigen::MatrixXf y = Eigen::MatrixXf::Zero(10, 1);
	for (int i = 0; i < 10; ++i) {
		for (int j = 0; j < 1; ++j) {
			float n = (rand() % 10) / 10;
			if (n >= 0.5) {
				y(i, j) = 1;
			}
		}
	}
	Variable* target = new Variable(&y);

	myNet net = myNet(1);
	Objective* criterion = new BinaryCrossEntropy();
	Optimizer* optimizer = new SGD(net.parameters());

	int EPOCH = 10;
	for (int epoch = 0; epoch < EPOCH; ++epoch) {
		optimizer->zero_grad();
		Variable* out = net(inputs);
		Variable* loss = (*criterion)(out, target);
		float acc = criterion->acc(out, target);
		cout << "iter: " << epoch <<" loss: " << loss->item() << " acc: " << 
			criterion->acc(out, target) << endl;
		loss->backward();
		optimizer->step();
	}
}