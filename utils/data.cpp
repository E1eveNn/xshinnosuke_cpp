#include "data.h"


MatrixType* batch_x_data = NULL;
MatrixType* batch_y_data = NULL;
Variable* batch_x = NULL;
Variable* batch_y = NULL;
pair<Variable*, Variable*> batch_data(batch_x, batch_y);


DataLoaderIterator::DataLoaderIterator(int length, int batch_size) {
	this->_index = length / batch_size;
}

DataLoaderIterator::DataLoaderIterator(MatrixType& data_matrix, MatrixType& target_matrix, 
	int batch_size, bool shuffle) {
	this->_data_matrix = data_matrix;
	this->_target_matrix = target_matrix;
	this->_index = 0;
	this->_length = data_matrix.rows();
	this->_batch_size = batch_size;
	this->_data_cols = data_matrix.cols();
	this->_target_cols = target_matrix.cols();
	this->sp_list = vector<long>(this->_length / batch_size);
	for (int i = 0, sp = 0; i < this->_length / batch_size; ++i, sp += batch_size) {
		sp_list[i] = sp;
	}
	if (shuffle) {
		std::random_device rd;
		std::mt19937 rng(rd());
		std::shuffle(sp_list.begin(), sp_list.end(), rng);
	}
}


DataLoaderIterator& DataLoaderIterator::operator ++() {
	this->_index++;
	return *this;
}

bool DataLoaderIterator::operator !=(const DataLoaderIterator& other) {
	return this->_index != other._index;
}

pair<Variable*, Variable*>& DataLoaderIterator::operator *() {
	long sp = this->sp_list[this->_index];
	int gap = (sp + this->_batch_size) >= this->_length ? (this->_length - sp - 1) : 
		this->_batch_size;
	batch_x_data = new MatrixType(this->_data_matrix.block(sp, 0, gap,
			this->_data_cols));
	batch_y_data = new MatrixType(this->_target_matrix.block(sp, 0, gap,
		this->_target_cols));
	if (batch_x == NULL) {
		batch_x = new Variable(batch_x_data);
		batch_data.first = batch_x;
	}
	else {
		delete batch_x->data;
		batch_x->data = batch_x_data;
	}

	if (batch_y == NULL) {
		batch_y = new Variable(batch_y_data);
		batch_data.second = batch_y;
	}
	else {
		delete batch_y->data;
		batch_y->data = batch_y_data;
	}
	return batch_data;
}


Dataset::Dataset(MatrixType& data_matrix, MatrixType& target_matrix) {
	if (data_matrix.rows() != target_matrix.rows()) {
		throw "data_matrix must has the same length of target_matrix!";
	}
	this->data_matrix = data_matrix;
	this->target_matrix = target_matrix;
}

Dataset::Dataset(MatrixType* data_matrix, MatrixType* target_matrix) {
	if (data_matrix->rows() != target_matrix->rows()) {
		throw "data_matrix must has the same length of target_matrix!";
	}
	this->data_matrix = *data_matrix;
	this->target_matrix = *target_matrix;
}

Dataset::Dataset(MatrixType* data_matrix) {
	this->data_matrix = *data_matrix;
}


DataLoader::DataLoader(Dataset& dataset, int batch_size, bool shuffle) {
	this->dataset = dataset;
	this->batch_size = batch_size;
	this->shuffle = shuffle;

}

DataLoaderIterator DataLoader::begin() {
	return DataLoaderIterator(this->dataset.data_matrix, this->dataset.target_matrix, 
		this->batch_size, this->shuffle);
}

DataLoaderIterator DataLoader::end() {
	return DataLoaderIterator(this->dataset.data_matrix.rows(), this->batch_size);
}


DataLoader::~DataLoader() {
	if (batch_x != NULL) {
		delete batch_x;
		batch_x = NULL;
		batch_x_data = NULL;
	}
	if (batch_y != NULL) {
		delete batch_y;
		batch_y = NULL;
		batch_y_data = NULL;
	}
}