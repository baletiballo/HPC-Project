#pragma once

#include <vector>
#include <random>
#include "ParallelStuff.cpp"

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
	int packets = 64;
	int packetSize;

	Conv(int f, int c1, int c2, int n, int s1, int s2) {
		num_filters = f;
		conv_size1 = c1;
		conv_size2 = c2;
		num_of_inputs = n;
		input_size1 = s1;
		input_size2 = s2;
		num_windows1 = input_size1 - conv_size1 + 1;
		num_windows2 = input_size2 - conv_size2 + 1;
		packetSize = (num_filters * num_of_inputs * num_windows1 * num_windows2) / packets;

		filters.resize(num_filters, vector<vector<float>>(conv_size1, vector<float>(conv_size2)));
		biases.resize(num_filters, 0.0);

		normal_distribution<float> distribution(0.0, 1.0);
		random_device dev;
		default_random_engine generator(dev());
		for (int cur_filter = 0; cur_filter < num_filters; cur_filter++) {
			for (int i = 0; i < conv_size1; i++) {
				for (int j = 0; j < conv_size2; j++) {
					filters[cur_filter][i][j] = distribution(generator) / (conv_size1 * conv_size2);
				}
			}
		}
	}

	void forwardJob(int packet) {
		for (int i = packet * packetSize; i < (packet + 1) * packetSize; i++) {
			int index1 = i / (num_windows1 * num_windows2);
			int index2 = (i / num_windows2) % num_windows1;
			int index3 = i % num_windows2;
			(*curr_output)[index1][index2][index3] = biases[index1 / num_of_inputs];

			//set output at i j for the input representation cur_featureMap when filter cur_filter is applied
			//matrix multiplication and summation
			for (int m = 0; m < conv_size1; m++) {
				for (int n = 0; n < conv_size2; n++) {
					(*curr_output)[index1][index2][index3] += (*curr_input)[index1 % num_of_inputs][index2 + m][index3 + n]
							* filters[index1 / num_of_inputs][m][n];
				}
			}
		}
		sem.V(1);
	}

	void forwardJobCleanup(int packet) {
		for (int i = packet * packetSize; i < num_filters * num_of_inputs * num_windows1 * num_windows2; i++) {
			int index1 = i / (num_windows1 * num_windows2);
			int index2 = (i / num_windows2) % num_windows1;
			int index3 = i % num_windows2;
			int filter = index1 / num_of_inputs;
			int featureMap = index1 % num_of_inputs;

			(*curr_output)[index1][index2][index3] = biases[filter];

			//set output at i j for the input representation cur_featureMap when filter cur_filter is applied
			//matrix multiplication and summation
			for (int m = 0; m < conv_size1; m++) {
				for (int n = 0; n < conv_size2; n++) {
					(*curr_output)[index1][index2][index3] += (*curr_input)[featureMap][index2 + m][index3 + n] * filters[filter][m][n];
				}
			}
		}
	}

	vector<vector<vector<float>>> forward(vector<vector<vector<float>>> &input) {
		vector<vector<vector<float>>> output(num_of_inputs * num_filters, vector<vector<float>>(num_windows1, vector<float>(num_windows2)));
		curr_output = &output;
		curr_input = &input;

		sem.set(0);
		for (int i = 0; i < packets; i++) {
			packaged_task<void()> job(bind(&forwardJob, this, i));
			pushJob(move(job));
		}
		if ((num_filters * num_of_inputs * num_windows1 * num_windows2) % packets != 0) {
			forwardJobCleanup(packets + 1);
		}
		sem.P(packets);

		return output;
	}

	void backpropJob(int packet) {
		for (int i = packet * packetSize; i < (packet + 1) * packetSize; i++) {
			int index1 = i / (num_windows1 * num_windows2);
			int index2 = (i / num_windows2) % num_windows1;
			int index3 = i % num_windows2;
			int filter = index1 / num_of_inputs;
			int featureMap = index1 % num_of_inputs;

			// per region
			//matrix multiplication and summation
			for (int m = 0; m < conv_size1; m++) {
				for (int n = 0; n < conv_size2; n++) {
					(*curr_filter_gradient)[filter][m][n] += (*curr_loss_gradient)[index1][index2][index3] * (*curr_input)[featureMap][index2 + m][index3 + n];
					(*curr_loss_input)[featureMap][index2 + m][index3 + n] += (*curr_loss_gradient)[index1][index2][index3] * filters[filter][m][n];
				}
			}

			(*curr_bias_gradient)[filter] += (*curr_loss_gradient)[index1][index2][index3];
		}
		sem.V(1);
	}

	void backpropJobCleanup(int packet) {
		for (int i = packet * packetSize; i < num_filters * num_of_inputs * num_windows1 * num_windows2; i++) {
			int index1 = i / (num_windows1 * num_windows2);
			int index2 = (i / num_windows2) % num_windows1;
			int index3 = i % num_windows2;
			int filter = index1 / num_of_inputs;
			int featureMap = index1 % num_of_inputs;

			// per region
			//matrix multiplication and summation
			for (int m = 0; m < conv_size1; m++) {
				for (int n = 0; n < conv_size2; n++) {
					(*curr_filter_gradient)[filter][m][n] += (*curr_loss_gradient)[index1][index2][index3] * (*curr_input)[featureMap][index2 + m][index3 + n];
					(*curr_loss_input)[featureMap][index2 + m][index3 + n] += (*curr_loss_gradient)[index1][index2][index3] * filters[filter][m][n];
				}
			}

			(*curr_bias_gradient)[filter] += (*curr_loss_gradient)[index1][index2][index3];
		}
	}

	tuple<vector<vector<vector<float>>>, vector<float>, vector<vector<vector<float>>>> backprop(vector<vector<vector<float>>> &loss_gradient,
			vector<vector<vector<float>>> &last_input) {
		vector<vector<vector<float>>> filter_gradient(num_filters, vector<vector<float>>(conv_size1, vector<float>(conv_size2, 0.0)));
		vector<float> bias_gradient(num_filters, 0.0);
		vector<vector<vector<float>>> loss_input(num_of_inputs, vector<vector<float>>(input_size1, vector<float>(input_size2, 0.0)));

		curr_filter_gradient = &filter_gradient;
		curr_bias_gradient = &bias_gradient;
		curr_loss_input = &loss_input;
		curr_loss_gradient = &loss_gradient;
		curr_input = &last_input;

		sem.set(0);
		for (int i = 0; i < packets; i++) {
			packaged_task<void()> job(bind(&backpropJob, this, i));
			pushJob(move(job));
		}
		if ((num_filters * num_of_inputs * num_windows1 * num_windows2) % packets != 0) {
			backpropJobCleanup(packets + 1);
		}
		sem.P(packets);

		return {filter_gradient, bias_gradient, loss_input};
	}
};