#include "ReLu.h"

const int reluSize1 = num_filters;
const int reluSize2 = imageSizeX_afterConvolution;
const int reluSize3 = imageSizeY_afterConvolution;

void ReLu(float t1 [reluSize1] [reluSize2] [reluSize3]) {
	for (unsigned i = 0; i < reluSize1; i++) {
		for (unsigned j = 0; j < reluSize2; j++) {
			for(unsigned k = 0; k < reluSize3; k++){
				
				if (t1[i][j][k+0] <= 0)
					t1[i][j][k+0] = 0;

			}
		}
	}
}

void ReLuPrime(float t1 [reluSize1] [reluSize2] [reluSize3], float t2  [reluSize1] [reluSize2] [reluSize3]) {
	for (unsigned i = 0; i < reluSize1; i++) {
		for (unsigned j = 0; j < reluSize2; j++) {
			for(unsigned k = 0; k < reluSize3; k++){
				
				if (t2[i][j][k] <= 0)
					t1[i][j][k] = 0;
				
			}
		}
	}
}