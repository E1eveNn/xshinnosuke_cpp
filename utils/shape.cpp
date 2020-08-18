#include "shape.h"


Shape::Shape(const initializer_list<int>& l) {
	this->size = 1;
	for (auto s = l.begin(); s != l.end(); ++s) {
		this->data.push_back(*s);
		this->size *= *s;
		this->ndim++;
	}
	this->ndim = this->data.size();
	this->initialize_flag = true;
}


Shape::Shape() {
	this->size = 0;
	this->ndim = 0;
	this->initialize_flag = false;
}

//Shape::Shape(const Shape& s) {
//	this->size = s.size;
//	this->data = s.data;
//	this->ndim = s.ndim;
//}

Shape Shape::slice(int sp, int ep) {
	return Shape({ this->data[sp] });
}

bool Shape::operator==(const Shape& other) {
	if (this->ndim != other.ndim || this->size != other.size)
		return false;
	for (int i = 0; i < this->ndim; ++i) {
		if (this->data[i] != other.data[i])
			return false;
	}
	return true;
}

Shape Shape::operator+(int a) {
	if (a <= 0) {
		return *this;
	}
	Shape ret(*this);
	ret.ndim++;
	ret.data.push_back(a);
	ret.size *= a;
	return ret;
}


int Shape::operator[](int dim) {
	if (dim >= this->ndim)
		throw "dim can't > ndim";
	return this->data[dim];
}