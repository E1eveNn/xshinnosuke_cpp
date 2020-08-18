# XShinnosuke_cpp: Deep Learning Framework
<div align=center>
	<img src="https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1597579280045&di=409d33924532df749524161e4c11f8b3&imgtype=0&src=http%3A%2F%2Fb-ssl.duitang.com%2Fuploads%2Fitem%2F201607%2F30%2F20160730144641_4UMvr.thumb.700_0.jpeg" width="400px" height="500px">
</div>

Xshinnosuke_cpp is the cpp version of [xshinnosuke](https://github.com/eLeVeNnN/xshinnosuke). 

As xshinnosuke_cpp choose [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) as matrix backend and [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) only supports for **Array** and **Matrix**, in other words, **the data in Eigen is less than 3 dimensions, so xshinnosuke_cpp just supports operations for 2-dimension datas**, such as *Linear(Linear in [Pytorch](https://pytorch.org/) or Dense in [Keras](https://keras.io/)), relu, sigmoid, batch normalization, etc*. 

For more *functional* or *Layers* details, such as Conv2D, max_pool2d, embedding, lstm, etc. Please refer to [xshinnosuke](https://github.com/eLeVeNnN/xshinnosuke).



Here are some features of XShinnosuke:

1. Based on **Eigen** and native to **C++**.

2. **Without** any other **3rd-party** deep learning library.

3. **Keras and Pytorch style API**, easy to start.

4. **Sequential** in Keras and Pytorch, **Model** in Keras and **Module** in Pytorch, all of them are supported in xshinnosuke_cpp.

5. Training and inference supports for both **dynamic graph** and **static graph**.

6. **Autograd** is supported .

   

## Getting started

### Dynamic Graph

```c++
#pragma once
#include<iostream>
#include "../layers/linear.h"
#include "../models.h"
#include "../nn/functional.h"
#include "../utils/data.h"
#include<ctime>
#include<vector>
using namespace std;

// define the network
class myNet : public Module {
// your diy network must be inherited from Module
public:
    // declare layers
	myNet(int n_classes) {
		this->fc1 = new Linear(10);
		this->relu1 = new ReLU();
		this->fc2 = new Linear(5);
		this->relu2 = new ReLU();
		this->fc3 = new Linear(n_classes);

	};
    // free the pointers
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
    // design forward flow
	Variable* forward(Variable* x) {
		Variable* out;
		out = (*(this->fc1))(x);
		out = (*(this->relu1))(out);
		out = (*(this->fc2))(out);
		out = (*(this->relu2))(out);
		out = (*(this->fc3))(out);
        // use funcational -> sigmoid
		out = sigmoid(out);
		return out;
	}
	
	Layer* fc1;
	Layer* relu1;
	Layer* fc2;
	Layer* relu2;
	Layer* fc3;
};
```

```c++
void run_dynamic() {
    // random generate training datas
    // generate training input datas
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
    // batch_size
	int BATCH_SIZE = 20;
    // xshinnosuke_cpp provides DataLoader for batch training/inference
	Dataset train_set = Dataset(x, y);
	DataLoader train_loader = DataLoader(train_set, BATCH_SIZE);
	// instantiate your diy network
	myNet net = myNet(1);
     // declare criterion
	Objective* criterion = new BinaryCrossEntropy();
    // declare optimizer, notice pass network's parameters to optimizer
	Optimizer* optimizer = new SGD(net.parameters());

	clock_t startTime, endTime;
	startTime = clock();
	int EPOCH = 10;
    // define training flow
    for (int epoch = 0; epoch < EPOCH; ++epoch) {
		for (auto it = train_loader.begin(); it != train_loader.end(); ++it) {
			Variable *inputs = (*it).first, *target = (*it).second;
            // at every epoch zero_grad's optimizer's trainable_variables' grad.
			optimizer->zero_grad();
            // forward
			Variable* out = net(inputs);
            // calculate loss
			Variable* loss = (*criterion)(out, target);
			float acc = criterion->calc_acc(out, target);
			cout << "iter: " << epoch << " loss: " << loss->item() << " acc: " <<
				acc << endl;
            // backward
			loss->backward();
            // update parameters
			optimizer->step();
		}
	}
	endTime = clock();
	cout << "The run time is: " <<
		(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
```



### Static Graph

```c++
#pragma once
#include "../layers/base.h"
#include "../layers/linear.h"
#include "../layers/activators.h"
#include "../models.h"
#include "../utils/shape.h"
#include<ctime>

void run_static() {
	int n_classes = 1;
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
	Sequential model = Sequential();
    // The first layer in Sequential must be specify input_shape(no need specify batch_size).
	model.add(new Linear(10, "null", false, "random", "zeros", Shape({ 10 })));
	model.add(new ReLU());
	model.add(new Linear(5));
	model.add(new ReLU());
	model.add(new Linear(n_classes));
	model.add(new Sigmoid());
	model.compile("sgd", "bce");
	clock_t startTime, endTime;
	startTime = clock();
	int BATCH_SIZE = 20;
	int EPOCH = 10;
	model.fit(&x, &y, BATCH_SIZE, EPOCH);
	endTime = clock();
	cout << "The run time is: " <<
		(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
}
```



## Installation

+ First download [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page).

+ ```git
  git clone https://github.com/eLeVeNnN/xshinnosuke_cpp.git
  ```

+ Include `Eigen` to xshinnosuke_cpp projects



## Demo

```c++
run demo.cpp
```

