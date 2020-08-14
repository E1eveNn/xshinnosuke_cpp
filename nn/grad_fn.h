#pragma once
#include "core.h"


void DenseBackward(Variable* outputs);
void ReLUBackward(Variable* outputs);
void SigmoidBackward(Variable* outputs);

// objectives
void BinaryCrossEntropyBackward(Variable* outputs);