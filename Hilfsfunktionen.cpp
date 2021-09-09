#include "Hilfsfunktionen.h"

void softmax(float t1 [num_classes]) {
	float sum = 0.0;
	
	for (unsigned i = 0; i < num_classes; i++) {
		t1[i] = exp(t1[i]);
		sum += t1[i];
	}

	for (unsigned i = 0; i < num_classes; i++) 
		t1[i] = t1[i] / sum;
}
