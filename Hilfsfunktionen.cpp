#include "Hilfsfunktionen.h"
#include <tgmath.h>

vector<float> flatten(vector<vector<vector<float>>> &t1) {
	vector<float> out(t1.size() * t1[0].size() * t1[0][0].size());
	for (unsigned i = 0; i < t1.size(); i++) {
		for (unsigned j = 0; j < t1[0].size(); j++) {
			for (unsigned k = 0; k < t1[0][0].size(); k++) {
				out[i * t1[0].size() * t1[0][0].size() + j * t1[0][0].size() + k] = t1[i][j][k];
			}
		}
	}
	return out;
}

vector<vector<vector<float>>> deflatten(vector<float> &t1, int s1, int s2, int s3) {
	vector<vector<vector<float>>> out(s1, vector<vector<float>>(s2, vector<float>(s3)));
	for (int i = 0; i < s1; i++) {
		for (int j = 0; j < s2; j++) {
			for (int k = 0; k < s3; k++) {
				out[i][j][k] = t1[i * s2 * s3 + j * s3 + k];
			}
		}
	}
	return out;
}

void softmax(vector<float> &t1, vector<float> &output) { //t1 and output have to have the same size
	float sum = 0.0;
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = exp(t1[i]);
		sum += output[i];
	}
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = output[i] / sum;
	}
}
