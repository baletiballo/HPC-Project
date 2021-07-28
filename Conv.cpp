#include "Conv.h"
#include <random>
#include "ParallelStuff.h"

Conv::Conv(int f, int c1, int c2, int n, int s1, int s2) {
	num_filters = f;
	conv_size1 = c1;
	conv_size2 = c2;
	num_of_inputs = n;
	input_size1 = s1;
	input_size2 = s2;
	num_windows1 = input_size1 - conv_size1 + 1;
	num_windows2 = input_size2 - conv_size2 + 1;
	packetSizeForw = (num_filters * num_of_inputs * num_windows1 * num_windows2) / packets;
	packetSizeBack = (num_of_inputs * num_windows1 * num_windows2) / packets;
	needForwCleanup = (num_filters * num_of_inputs * num_windows1 * num_windows2) % packets != 0;
	needBackCleanup=(num_of_inputs * num_windows1 * num_windows2) % packets != 0;

	filters.resize(num_filters, vector<vector<float>>(conv_size1, vector<float>(conv_size2)));
	biases.resize(num_filters, 0.0);

	output.resize(num_of_inputs * num_filters, vector<vector<float>>(num_windows1, vector<float>(num_windows2)));

	filter_gradient.resize(num_filters, vector<vector<float>>(conv_size1, vector<float>(conv_size2, 0.0)));
	bias_gradient.resize(num_filters, 0.0);
	loss_input.resize(num_of_inputs, vector<vector<float>>(input_size1, vector<float>(input_size2, 0.0)));

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

/**
 * Forward
 *
 * @param inputP
 * @return
 */
void Conv::forward(vector<vector<vector<float>>> &inputP) {
	input = &inputP;
	for (int cur_filter = 0; cur_filter < num_filters; cur_filter++) { //per filter
		for (int cur_featureMap = 0; cur_featureMap < num_of_inputs; cur_featureMap++) { //per input
			for (int i = 0; i < num_windows1; i++) {
				//per region
				for (int j = 0; j < num_windows2; j++) {
					// per region

					output[cur_featureMap + num_of_inputs * cur_filter][i][j] = biases[cur_filter];

					//set output at i j for the input representation cur_featureMap when filter cur_filter is applied
					//matrix multiplication and summation
					for (int m = 0; m < conv_size1; m++) {
						for (int n = 0; n < conv_size2; n++) {
							output[cur_featureMap + num_of_inputs * cur_filter][i][j] += (*input)[cur_featureMap][i + m][j + n] * filters[cur_filter][m][n];
						}
					}
				}
			}
		}
	}
}

/**
 * Call this after every Batch, addition of the gradients throughout a Batch is now done directly in backprop
 *
 */
void Conv::cleanup() {
	for (int cur_filter = 0; cur_filter < num_filters; cur_filter++) {
		bias_gradient[cur_filter] = 0;
		for (int i = 0; i < conv_size1; i++) {
			for (int j = 0; j < conv_size2; j++) {
				filter_gradient[cur_filter][i][j] = 0;
			}
		}
	}
}

/**
 * Backprop (notice: gradients are getting already added here, since filter_gradient and bias_gradient only get reset to zero after a cleanup() call and are only needed after a batch)
 *
 * @param loss_gradientP
 * @return
 */
void Conv::backprop(vector<vector<vector<float>>> &loss_gradientP) {
	loss_gradient = &loss_gradientP;

	//zero the loss Input, since the same method to just add them all together cannot be applied here
	for (int cur_featureMap = 0; cur_featureMap < num_of_inputs; cur_featureMap++) {
		for (int i = 0; i < input_size1; i++) {
			for (int j = 0; j < input_size2; j++) {
				loss_input[cur_featureMap][i][j] = 0;
			}
		}
	}

	for (int cur_filter = 0; cur_filter < num_filters; cur_filter++) { //per filter
		for (int cur_featureMap = 0; cur_featureMap < num_of_inputs; cur_featureMap++) { //per input

			for (int i = 0; i < num_windows1; i++) {
				//per region
				for (int j = 0; j < num_windows2; j++) {
					// per region
					//matrix multiplication and summation
					for (int m = 0; m < conv_size1; m++) { //unroll? 
						for (int n = 0; n < conv_size2; n++) {
							filter_gradient[cur_filter][m][n] += (*loss_gradient)[cur_featureMap + num_of_inputs * cur_filter][i][j]
									* (*input)[cur_featureMap][i + m][j + n];
							loss_input[cur_featureMap][i + m][j + n] += (*loss_gradient)[cur_featureMap + num_of_inputs * cur_filter][i][j]
									* filters[cur_filter][m][n];
						}
					}

					bias_gradient[cur_filter] += (*loss_gradient)[cur_featureMap * num_filters + cur_filter][i][j];
				}
			}
		}
	}
}

/**
 * Description
 *
 * @param packet
 * @return
 */
void Conv::forwardJob(int packet) {
	for (int index = packet * packetSizeForw; index < (packet + 1) * packetSizeForw; index++) {
		int cur_featureMap = index / (num_windows1 * num_windows2 * num_filters);
		int cur_filter = (index / (num_windows1 * num_windows2)) % num_filters;
		int i = (index / num_windows2) % num_windows1;
		int j = index % num_windows2;

		output[cur_featureMap + num_of_inputs * cur_filter][i][j] = biases[cur_filter];

		//set output at i j for the input representation cur_featureMap when filter cur_filter is applied
		//matrix multiplication and summation
		for (int m = 0; m < conv_size1; m++) {
			for (int n = 0; n < conv_size2; n++) {
				output[cur_featureMap + num_of_inputs * cur_filter][i][j] += (*input)[cur_featureMap][i + m][j + n] * filters[cur_filter][m][n];
			}
		}
	}
	sem.V(1);
}

/**
 * Description
 *
 * @param packet
 * @return
 */
void Conv::forwardJobCleanup(int packet) {
	for (int index = packet * packetSizeForw; index < num_filters * num_of_inputs * num_windows1 * num_windows2; index++) {
		int cur_featureMap = index / (num_windows1 * num_windows2 * num_filters);
		int cur_filter = (index / (num_windows1 * num_windows2)) % num_filters;
		int i = (index / num_windows2) % num_windows1;
		int j = index % num_windows2;

		output[cur_featureMap + num_of_inputs * cur_filter][i][j] = biases[cur_filter];

		//set output at i j for the input representation cur_featureMap when filter cur_filter is applied
		//matrix multiplication and summation
		for (int m = 0; m < conv_size1; m++) {
			for (int n = 0; n < conv_size2; n++) {
				output[cur_featureMap + num_of_inputs * cur_filter][i][j] += (*input)[cur_featureMap][i + m][j + n] * filters[cur_filter][m][n];
			}
		}
	}
}

/**
 * Description
 *
 * @param inputP
 * @return
 */
void Conv::forward_par(vector<vector<vector<float>>> &inputP) {
	input = &inputP;

	sem.set(0);
	pool.setConv(*this);
	pool.setTask(1);
	for (int i = 0; i < packets; i++) {
		pushJob(i);
	}
	if (needForwCleanup) {
		forwardJobCleanup(packets + 1);
	}
	sem.P(packets);
}

void Conv::backpropJob(int packet) {
	for (int index = packet * packetSizeBack; index < (packet + 1) * packetSizeBack; index++) {
		int cur_featureMap = index / (num_windows1 * num_windows2);
		int i = (index / num_windows2) % num_windows1;
		int j = index % num_windows2;
		loss_input[cur_featureMap][i][j] = 0;

		for (int cur_filter = 0; cur_filter < num_filters; cur_filter++) { //per filter
			for (int m = 0; m < min(i + 1, conv_size1); m++) { //unroll?
				for (int n = 0; n < min(j + 1, conv_size2); n++) {
					loss_input[cur_featureMap][i][j] += (*loss_gradient)[cur_featureMap + num_of_inputs * cur_filter][i - m][j - n] * filters[cur_filter][m][n];
				}
			}
		}
	}

	for (int cur_filter = packet; cur_filter < num_filters; cur_filter += packets) {

		for (int cur_featureMap = 0; cur_featureMap < num_of_inputs; cur_featureMap++) { //per input
			for (int i = 0; i < num_windows1; i++) {
				//per region
				for (int j = 0; j < num_windows2; j++) {
					// per region
					//matrix multiplication and summation
					for (int m = 0; m < conv_size1; m++) { //unroll?
						for (int n = 0; n < conv_size2; n++) {
							filter_gradient[cur_filter][m][n] += (*loss_gradient)[cur_featureMap + num_of_inputs * cur_filter][i][j]
									* (*input)[cur_featureMap][i + m][j + n];
						}
					}
					bias_gradient[cur_filter] += (*loss_gradient)[cur_featureMap * num_filters + cur_filter][i][j];
				}
			}
		}
	}

	sem.V(1);
}

/**
 * Description
 *
 * @param packet
 * @return
 */
void Conv::backpropJobCleanup(int packet) {
	for (int index = packet * packetSizeBack; index < num_of_inputs * num_windows1 * num_windows2; index++) {
		int cur_featureMap = index / (num_windows1 * num_windows2);
		int i = (index / num_windows2) % num_windows1;
		int j = index % num_windows2;
		loss_input[cur_featureMap][i][j] = 0;

		for (int cur_filter = 0; cur_filter < num_filters; cur_filter++) { //per filter
			for (int m = 0; m < min(i + 1, conv_size1); m++) { //unroll?
				for (int n = 0; n < min(j + 1, conv_size2); n++) {
					loss_input[cur_featureMap][i][j] += (*loss_gradient)[cur_featureMap + num_of_inputs * cur_filter][i - m][j - n] * filters[cur_filter][m][n];
				}
			}
		}
	}

	for (int cur_filter = packet; cur_filter < num_filters; cur_filter += packets) {

		for (int cur_featureMap = 0; cur_featureMap < num_of_inputs; cur_featureMap++) { //per input
			for (int i = 0; i < num_windows1; i++) {
				//per region
				for (int j = 0; j < num_windows2; j++) {
					// per region
					//matrix multiplication and summation
					for (int m = 0; m < conv_size1; m++) { //unroll?
						for (int n = 0; n < conv_size2; n++) {
							filter_gradient[cur_filter][m][n] += (*loss_gradient)[cur_featureMap + num_of_inputs * cur_filter][i][j]
									* (*input)[cur_featureMap][i + m][j + n];
						}
					}
					bias_gradient[cur_filter] += (*loss_gradient)[cur_featureMap * num_filters + cur_filter][i][j];
				}
			}
		}
	}
}

/**
 * Description
 *
 * @param loss_gradientP
 * @return
 */
void Conv::backprop_par(vector<vector<vector<float>>> &loss_gradientP) {
	loss_gradient = &loss_gradientP;

	sem.set(0);
	pool.setConv(*this);
	pool.setTask(2);
	for (int i = 0; i < packets; i++) {
		pushJob(i);
	}
	if (needBackCleanup) {
		backpropJobCleanup(packets + 1);
	}
	sem.P(packets);
}
