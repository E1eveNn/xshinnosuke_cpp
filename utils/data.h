#pragma once
#include "../nn/core.h"
#include<random>
using namespace std;

extern MatrixType* batch_x_data;
extern MatrixType* batch_y_data;
extern Variable* batch_x;
extern Variable* batch_y;
extern pair<Variable*, Variable*> batch_data;



class DataLoaderIterator {
private:
	int _index;
	int _ep;
	int _length;
	int _data_cols;
	int _target_cols;
	int _batch_size;
	MatrixType _data_matrix;
	MatrixType _target_matrix;
	vector<long> sp_list;
public:
	DataLoaderIterator(int length, int batch_size);
	DataLoaderIterator(MatrixType& data_matrix, MatrixType& target_matrix, 
		int batch_size, bool shuffle);
	DataLoaderIterator& operator ++();
	bool operator !=(const DataLoaderIterator& other);
	pair<Variable*, Variable*>& operator *();
};


class Dataset {
public:
	Dataset() {};

	Dataset(MatrixType& data_matrix, MatrixType& target_matrix);
	Dataset(MatrixType* data_matrix, MatrixType* target_matrix);
	Dataset(MatrixType* data_matrix);
	MatrixType data_matrix;
	MatrixType target_matrix;
};


class DataLoader {
public:

	DataLoader() {};
	DataLoader(Dataset& dataset, int batch_size = 32, bool shuffle = true);
	~DataLoader();
	DataLoaderIterator begin();
	DataLoaderIterator end();

private:
	Dataset dataset;
	int batch_size;
	bool shuffle;
};


