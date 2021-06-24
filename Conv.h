/*
 * Conv.h
 *
 *  Created on: 24.06.2021
 *      Author: Stefan
 */

#ifndef CONV_H_
#define CONV_H_

#include <vector>

using namespace std;

class Conv {
public:
	int num_filters;
	int num_of_inputs, input_size1, input_size2;
	int conv_size1, conv_size2;
	int num_windows1, num_windows2;
	vector<vector<vector<float>>> filters;
	vector<float> biases;
	vector<vector<vector<float>>> *curr_input = nullptr;
	vector<vector<vector<float>>> *curr_output = nullptr;
	vector<vector<vector<float>>> *curr_loss_gradient = nullptr;
	vector<vector<vector<float>>> *curr_filter_gradient = nullptr;
	vector<float> *curr_bias_gradient = nullptr;
	vector<vector<vector<float>>> *curr_loss_input = nullptr;
	int packets = 12;
	int packetSize;

	Conv(int f, int c1, int c2, int n, int s1, int s2);

	void forwardJob(int packet);

	void forwardJobCleanup(int packet);

	vector<vector<vector<float>>> forward(vector<vector<vector<float>>> &input);

	void backpropJob(int packet);

	void backpropJobCleanup(int packet);

	tuple<vector<vector<vector<float>>>, vector<float>, vector<vector<vector<float>>>> backprop(vector<vector<vector<float>>> &loss_gradient,
			vector<vector<vector<float>>> &last_input);
};

#endif /* CONV_H_ */
