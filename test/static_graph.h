#pragma once
#include "../layers/base.h"
#include "../layers/linear.h"
#include "../layers/activators.h"
#include "../models.h"
#include "../utils/shape.h"
#include<ctime>

void run_static() {
	int n_classes = 1;
	int samples = 1000;
	int features = 30;
	Eigen::MatrixXf x = Eigen::MatrixXf::Random(samples, features);
	Eigen::MatrixXf y = Eigen::MatrixXf::Zero(samples, 1);
	for (int i = 0; i < samples; ++i) {
		for (int j = 0; j < 1; ++j) {
			float n = (rand() % 10) / 10;
			if (n >= 0.5) {
				y(i, j) = 1;
			}
		}
	}
	Sequential model = Sequential();
	model.add(new Linear(10, "null", false, "random", "zeros", Shape({ features })));
	model.add(new ReLU());
	model.add(new Linear(5));
	model.add(new ReLU());
	model.add(new Linear(n_classes));
	model.add(new Sigmoid());
	model.compile("sgd", "bce");
	clock_t startTime, endTime;
	startTime = clock();
	int BATCH_SIZE = 100;
	int EPOCH = 20;
	model.fit(&x, &y, BATCH_SIZE, EPOCH);
	endTime = clock();
	cout << "The run time is: " <<
		(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
}