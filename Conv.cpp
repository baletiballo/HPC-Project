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
	packetSize = (num_filters * num_of_inputs * num_windows1 * num_windows2) / packets;

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
				loss_input[cur_featureMap][i][j]=0;
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
					for (int m = 0; m < conv_size1; m++) {
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

///**
// * Description
// *
// * @param packet
// * @return
//*/
//void Conv::forwardJob(int packet) {
//	for (int i = packet * packetSize; i < (packet + 1) * packetSize; i++) {
//		int index1 = i / (num_windows1 * num_windows2);
//		int index2 = (i / num_windows2) % num_windows1;
//		int index3 = i % num_windows2;
//		(*curr_output)[index1][index2][index3] = biases[index1 / num_of_inputs];
//
//		//set output at i j for the input representation cur_featureMap when filter cur_filter is applied
//		//matrix multiplication and summation
//		for (int m = 0; m < conv_size1; m++) {
//			for (int n = 0; n < conv_size2; n++) {
//				(*curr_output)[index1][index2][index3] += (*curr_input)[index1 % num_of_inputs][index2 + m][index3 + n] * filters[index1 / num_of_inputs][m][n];
//			}
//		}
//	}
//	sem.V(1);
//}
//
///**
// * Description
// *
// * @param packet
// * @return
//*/
//void Conv::forwardJobCleanup(int packet) {
//	for (int i = packet * packetSize; i < num_filters * num_of_inputs * num_windows1 * num_windows2; i++) {
//		int index1 = i / (num_windows1 * num_windows2);
//		int index2 = (i / num_windows2) % num_windows1;
//		int index3 = i % num_windows2;
//		int filter = index1 / num_of_inputs;
//		int featureMap = index1 % num_of_inputs;
//
//		(*curr_output)[index1][index2][index3] = biases[filter];
//
//		//set output at i j for the input representation cur_featureMap when filter cur_filter is applied
//		//matrix multiplication and summation
//		for (int m = 0; m < conv_size1; m++) {
//			for (int n = 0; n < conv_size2; n++) {
//				(*curr_output)[index1][index2][index3] += (*curr_input)[featureMap][index2 + m][index3 + n] * filters[filter][m][n];
//			}
//		}
//	}
//}
//
///**
// * Description
// *
// * @param input
// * @return
//*/
//vector<vector<vector<float>>> Conv::forward(vector<vector<vector<float>>> &input) {
//	vector<vector<vector<float>>> output(num_of_inputs * num_filters, vector<vector<float>>(num_windows1, vector<float>(num_windows2)));
//	curr_output = &output;
//	curr_input = &input;
//
//	sem.set(0);
//	pool.setConv(*this);
//		pool.setTask(1);
//	for (int i = 0; i < packets; i++) {
//		pushJob(i);
//	}
//	if ((num_filters * num_of_inputs * num_windows1 * num_windows2) % packets != 0) {
//		forwardJobCleanup(packets + 1);
//	}
//	sem.P(packets);
//
//	return output;
//}
//
//void Conv::backpropJob(int packet) {
//	for (int i = packet * packetSize; i < (packet + 1) * packetSize; i++) {
//		int index1 = i / (num_windows1 * num_windows2);
//		int index2 = (i / num_windows2) % num_windows1;
//		int index3 = i % num_windows2;
//		int filter = index1 / num_of_inputs;
//		int featureMap = index1 % num_of_inputs;
//
//		// per region
//		//matrix multiplication and summation
//		for (int m = 0; m < conv_size1; m++) {
//			for (int n = 0; n < conv_size2; n++) {
//				(*curr_filter_gradient)[filter][m][n] += (*curr_loss_gradient)[index1][index2][index3] * (*curr_input)[featureMap][index2 + m][index3 + n];
//				(*curr_loss_input)[featureMap][index2 + m][index3 + n] += (*curr_loss_gradient)[index1][index2][index3] * filters[filter][m][n];
//			}
//		}
//
//		(*curr_bias_gradient)[filter] += (*curr_loss_gradient)[index1][index2][index3];
//	}
//	sem.V(1);
//}
//
///**
// * Description
// *
// * @param packet
// * @return
//*/
//void Conv::backpropJobCleanup(int packet) {
//	for (int i = packet * packetSize; i < num_filters * num_of_inputs * num_windows1 * num_windows2; i++) {
//		int index1 = i / (num_windows1 * num_windows2);
//		int index2 = (i / num_windows2) % num_windows1;
//		int index3 = i % num_windows2;
//		int filter = index1 / num_of_inputs;
//		int featureMap = index1 % num_of_inputs;
//
//		// per region
//		//matrix multiplication and summation
//		for (int m = 0; m < conv_size1; m++) {
//			for (int n = 0; n < conv_size2; n++) {
//				(*curr_filter_gradient)[filter][m][n] += (*curr_loss_gradient)[index1][index2][index3] * (*curr_input)[featureMap][index2 + m][index3 + n];
//				(*curr_loss_input)[featureMap][index2 + m][index3 + n] += (*curr_loss_gradient)[index1][index2][index3] * filters[filter][m][n];
//			}
//		}
//
//		(*curr_bias_gradient)[filter] += (*curr_loss_gradient)[index1][index2][index3];
//	}
//}
//
///**
// * Description
// *
// * @param loss_gradient
// * @param last_input
// * @return
//*/
//tuple<vector<vector<vector<float>>>, vector<float>, vector<vector<vector<float>>>> Conv::backprop(vector<vector<vector<float>>> &loss_gradient,
//		vector<vector<vector<float>>> &last_input) {
//	vector<vector<vector<float>> > filter_gradient(num_filters, vector<vector<float>>(conv_size1, vector<float>(conv_size2, 0.0)));
//	vector<float> bias_gradient(num_filters, 0.0);
//	vector<vector<vector<float>> > loss_input(num_of_inputs, vector<vector<float>>(input_size1, vector<float>(input_size2, 0.0)));
//
//	curr_filter_gradient = &filter_gradient;
//	curr_bias_gradient = &bias_gradient;
//	curr_loss_input = &loss_input;
//	curr_loss_gradient = &loss_gradient;
//	curr_input = &last_input;
//
//	sem.set(0);
//	pool.setConv(*this);
//	pool.setTask(2);
//	for (int i = 0; i < packets; i++) {
//		pushJob(i);
//	}
//	if ((num_filters * num_of_inputs * num_windows1 * num_windows2) % packets != 0) {
//		backpropJobCleanup(packets + 1);
//	}
//	sem.P(packets);
//
//	return {filter_gradient, bias_gradient, loss_input};
//}
