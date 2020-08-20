#pragma once
#include "core.h"
#include "../utils/shape.h"


class Initializer {
public:
	virtual Variable* operator()(const Shape& shape) = 0;
	virtual Variable* operator()(int rows, int cols) = 0;
	//virtual Variable* operator()(int length) = 0;
	vector<int>& decompose_size(Shape& shape);
};


class Normal : public Initializer {
	Normal(float mean = 0.0, float std = 0.1);
};


class Zeros : public Initializer {
	virtual Variable* operator()(const Shape& shape);
	virtual Variable* operator()(int rows, int cols);
	//virtual Variable* operator()(int length);
};


class Ones : public Initializer {
	virtual Variable* operator()(const Shape& shape);
	virtual Variable* operator()(int rows, int cols);
	//virtual Variable* operator()(int length);
};

class Random : public Initializer {
	virtual Variable* operator()(const Shape& shape);
	virtual Variable* operator()(int rows, int cols);
};

class Binominal :public Initializer {
	virtual Variable* operator()(const Shape& shape);
	virtual Variable* operator()(int rows, int cols);
};

Initializer* get_initializer(const string& name);