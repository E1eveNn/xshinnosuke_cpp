# XShinnosuke_cpp: Deep Learning Framework
<div align=center>
	<img src="https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1597579280045&di=409d33924532df749524161e4c11f8b3&imgtype=0&src=http%3A%2F%2Fb-ssl.duitang.com%2Fuploads%2Fitem%2F201607%2F30%2F20160730144641_4UMvr.thumb.700_0.jpeg" width="400px" height="500px">
</div>

Xshinnosuke_cpp is the cpp version of [xshinnosuke](https://github.com/eLeVeNnN/xshinnosuke). 

As xshinnosuke_cpp choose [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) as matrix backend and [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) only supports for **Array** and **Matrix**, in other words, **the data in Eigen is less than 3 dimensions, so xshinnosuke_cpp just supports operations for 2-dimension datas**, such as *Linear(Linear in [Pytorch](https://pytorch.org/) or Dense in [Keras](https://keras.io/)), relu, sigmoid, batch normalization, etc*. 

For more *functional* or *Layers* details, such as Conv2D, max_pool2d, embedding, lstm, etc. Please refer to [xshinnosuke](https://github.com/eLeVeNnN/xshinnosuke).



Here are some features of Shinnosuke:

1. Based on **Eigen** and native to **C++**.

2. **Without** any other **3rd-party** deep learning library.

3. **Keras and Pytorch style API**, easy to start.

4. **Sequential** in Keras and Pytorch, **Model** in Keras and **Module** in Pytorch, all of them are supported in xshinnosuke_cpp.

5. Training and inference supports for both **dynamic graph** and **static graph**.

6. **Autograd** is supported .

   

## Getting started

```c++
#include<iostream>
#include "layers/linear.h"
#include "models.h"
#include "nn/functional.h"
#include "nn/objectives.h"
#include "nn/optimizers.h"
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
int main() {
    // random generate training datas
    // generate training input datas
	Eigen::MatrixXf x = Eigen::MatrixXf::Random(10, 10);
	Variable* inputs = new Variable(&x);
	inputs->requires_grad_(true);
    // generate training input targets
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
	
    // instantiate your diy network
	myNet net = myNet(1);
    // declare criterion
	Objective* criterion = new BinaryCrossEntropy();
    // declare optimizer, notice pass network's parameters to optimizer
	Optimizer* optimizer = new SGD(net.parameters());
	// define training EPOCH
	int EPOCH = 10;
    // define training flow
	for (int epoch = 0; epoch < EPOCH; ++epoch) {
        // at every epoch zero_grad's optimizer's trainable_variables' grad.
		optimizer->zero_grad();
        // forward
		Variable* out = net(inputs);
        // calculate loss
		Variable* loss = (*criterion)(out, target);
        // calculate accuracy
		float acc = criterion->acc(out, target);
		cout << "iter: " << epoch <<" loss: " << loss->item() << " acc: " << 
			criterion->acc(out, target) << endl;
        // backward
		loss->backward();
        // update parameters
		optimizer->step();
	}
}

// result
iter: 0 loss: 1.25291 acc: 0.2
iter: 1 loss: 1.00379 acc: 0.2
iter: 2 loss: 0.702379 acc: 0.4
iter: 3 loss: 0.484556 acc: 0.7
iter: 4 loss: 0.347854 acc: 1
iter: 5 loss: 0.264427 acc: 1
iter: 6 loss: 0.201489 acc: 1
iter: 7 loss: 0.138188 acc: 1
iter: 8 loss: 0.101462 acc: 1
iter: 9 loss: 0.0720882 acc: 1
```

## Installation

+ First download [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page).

+ ```git
  git clone https://github.com/eLeVeNnN/xshinnosuke_cpp.git
  ```

+ Include `Eigen` to xshinnosuke_cpp projects



## Demo

```c++
run test.cpp
```

