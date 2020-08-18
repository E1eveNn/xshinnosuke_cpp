#pragma once
#include<iostream>
#include "../layers/linear.h"
#include "../models.h"
#include "../nn/functional.h"
#include "../utils/data.h"
#include<ctime>
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
	~myNet() {
		delete this->fc1;
		delete this->relu1;
		delete this->fc2;
		delete this->relu2;
		delete this->fc3;
		this->fc1 = NULL;
		this->relu1 = NULL;
		this->fc2 = NULL;
		this->relu2 = NULL;
		this->fc3 = NULL;
	}
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



void run_dynamic() {
	Eigen::MatrixXf x = Eigen::MatrixXf::Random(100, 10);
	Eigen::MatrixXf y = Eigen::MatrixXf::Zero(100, 1);
	for (int i = 0; i < 10; ++i) {
		for (int j = 0; j < 1; ++j) {
			float n = (rand() % 10) / 10;
			if (n >= 0.5) {
				y(i, j) = 1;
			}
		}
	}
	int BATCH_SIZE = 20;
	Dataset train_set = Dataset(x, y);
	DataLoader train_loader = DataLoader(train_set, BATCH_SIZE);

	myNet net = myNet(1);
	Objective* criterion = new BinaryCrossEntropy();
	Optimizer* optimizer = new SGD(net.parameters());

	clock_t startTime, endTime;
	startTime = clock();
	int EPOCH = 10;
	for (int epoch = 0; epoch < EPOCH; ++epoch) {
		for (auto it = train_loader.begin(); it != train_loader.end(); ++it) {
			Variable *inputs = (*it).first, *target = (*it).second;
			optimizer->zero_grad();
			Variable* out = net(inputs);
			Variable* loss = (*criterion)(out, target);
			float acc = criterion->calc_acc(out, target);
			cout << "iter: " << epoch << " loss: " << loss->item() << " acc: " <<
				acc << endl;
			loss->backward();
			optimizer->step();
		}
	}
	endTime = clock();
	cout << "The run time is: " <<
		(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
}