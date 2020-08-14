#pragma once
#include "core.h"
#include "grad_fn.h"
#include "toolkit.h"

class Objective {
	Variable* operator()(Variable* y_pred, Variable* y_true);
	virtual Variable* forward(Variable* y_pred, Variable* y_true) = 0;
	virtual float acc(Variable* y_pred, Variable* y_true) = 0;
};


class BinaryCrossEntropy : Objective {
	virtual Variable* forward(Variable* y_pred, Variable* y_true);
	virtual float acc(Variable* y_pred, Variable* y_true);
};


