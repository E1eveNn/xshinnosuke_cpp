#pragma once
#include "core.h"

// math operations
void AddBackward(Variable* outputs);



void DenseBackward(Variable* outputs);
void ReLUBackward(Variable* outputs);
void SigmoidBackward(Variable* outputs);

// objectives
void BinaryCrossEntropyBackward(Variable* outputs);