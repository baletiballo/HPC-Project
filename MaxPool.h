/*
 * MaxPool.h
 *
 *  Created on: 24.06.2021
 *      Author: Stefan
 */

#ifndef MAXPOOL_H_
#define MAXPOOL_H_

#include <vector>

using namespace std;

class MaxPool {
public:
	int num_of_inputs, input_size1, input_size2;
	int window, stride;
	int output_size1, output_size2;
	vector<vector<vector<float>>> *curr_input = nullptr;
	vector<vector<vector<float>>> *curr_output = nullptr;
	vector<vector<vector<float>>> *curr_loss_gradient = nullptr;
	vector<vector<vector<float>>> *curr_loss_input = nullptr;
	int packets = 12;
	int packetSize;

	MaxPool(int w, int s, int n, int s1, int s2);

	void forwardJob(int packet);

	void forwardJobCleanup(int packet);

	vector<vector<vector<float>>> forward(vector<vector<vector<float>>> &input);

	void backpropJob(int packet);

	void backpropJobCleanup(int packet);

	vector<vector<vector<float>>> backprop(vector<vector<vector<float>>> &loss_gradient, vector<vector<vector<float>>> &last_input);
};

#endif /* MAXPOOL_H_ */
