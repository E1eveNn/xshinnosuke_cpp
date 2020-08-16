#pragma once
#include "functional.h"
#include "../utils/toolkit.h"


class Objective {
public:
	Variable* operator()(Variable* y_pred, Variable* y_true);
	virtual Variable* forward(Variable* y_pred, Variable* y_true) = 0;
	virtual float acc(Variable* y_pred, Variable* y_true) = 0;
};


class BinaryCrossEntropy : public Objective {
public:
	BinaryCrossEntropy();
	virtual Variable* forward(Variable* y_pred, Variable* y_true);
	virtual float acc(Variable* y_pred, Variable* y_true);
};


