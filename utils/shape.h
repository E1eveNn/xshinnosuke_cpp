#pragma once
#include <initializer_list>
#include<vector>
#include<iostream>
using namespace std;


class Shape {
public:
	friend ostream& operator<<(ostream& os, const Shape& t) {
		os << '(';
		for (int i = 0; i < t.ndim; ++i)
		{
			os << t.data[i];
			if(t.ndim == 1 or i != t.ndim - 1)
				os << ',';
		}
		os << ')' << endl;
		return os;
	}
	Shape();
	Shape(const initializer_list<int>& l);
	// Shape(const Shape& s);
	

	bool operator==(const Shape& other);
	Shape operator+(int a);
	int operator[](int dim);

	// record data size
	int size;

	vector<int> data;
	// record how many dimensions
	int ndim;

};