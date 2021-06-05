/*
 * cnn.cpp
 *
 *  Created on: 03.05.2021
 *      Author: Stefan, Hannah, Silas
 */

#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <random>
#include <tuple>
#include <float.h>
#include <chrono>
#include <ctime> 

using namespace std;

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

vector<vector<float>> transpose(vector<float> &t1) {
	vector<vector<float>> output(1, vector<float>(t1.size()));
	for (unsigned i = 0; i < t1.size(); i++) {
		output[0][i] = t1[i];
	}
	return output;
}

vector<vector<float>> transpose(vector<vector<float>> &t1) {
	vector<vector<float>> output(t1[0].size(), vector<float>(t1.size()));
	for (unsigned i = 0; i < t1.size(); i++) {
		for (unsigned j = 0; j < t1[0].size(); j++) {
			output[j][i] = t1[i][j];
		}
	}
	return output;
}

vector<vector<float>> addDim(vector<float> &t1) {
	vector<vector<float>> output(t1.size(), vector<float>(1));
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i][0] = t1[i];
	}
	return output;
}

vector<float> copy(vector<float> &t1) {
	vector<float> output(t1.size());
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = t1[i];
	}
	return output;
}

vector<float> softmax(vector<float> &t1) {
	float sum = 0.0;
	vector<float> res(t1.size());
	for (unsigned i = 0; i < t1.size(); i++) {
		res[i] = exp(t1[i]);
		sum += res[i];
	}
	for (unsigned i = 0; i < t1.size(); i++) {
		res[i] = res[i] / sum;
	}
	return res;
}

void ReLu(vector<float> &t1) {
	for (unsigned i = 0; i < t1.size(); i++) {
		if (t1[i] <= 0) {
			t1[i] = 0;
		}
	}
}

void ReLu(vector<vector<float>> &t1) {
	for (unsigned i = 0; i < t1.size(); i++) {
		ReLu(t1[i]);
	}
}

void ReLu(vector<vector<vector<float>>> &t1) {
	for (unsigned i = 0; i < t1.size(); i++) {
		ReLu(t1[i]);
	}
}

void ReLu(vector<vector<vector<vector<float>>>> &t1) {
	for (unsigned i = 0; i < t1.size(); i++) {
		ReLu(t1[i]);
	}
}

void ReLuPrime(vector<float> &t1, vector<float> &t2) {
	for (unsigned i = 0; i < t1.size(); i++) {
		if (t2[i] <= 0) {
			t1[i] = 0;
		}
	}
}

void ReLuPrime(vector<vector<float>> &t1, vector<vector<float>> &t2) {
	for (unsigned i = 0; i < t1.size(); i++) {
		ReLuPrime(t1[i], t2[i]);
	}
}

void ReLuPrime(vector<vector<vector<float>>> &t1, vector<vector<vector<float>>> &t2) {
	for (unsigned i = 0; i < t1.size(); i++) {
		ReLuPrime(t1[i], t2[i]);
	}
}

void ReLuPrime(vector<vector<vector<vector<float>>>> &t1, vector<vector<vector<vector<float>>>> &t2) {
	for (unsigned i = 0; i < t1.size(); i++) {
		ReLuPrime(t1[i], t2[i]);
	}
}

vector<float> multiply(vector<float> t1, float t2) {
	vector<float> output(t1.size(), 0.0);
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] += t1[i] * t2;
	}
	return output;
}

vector<vector<float>> multiply(vector<vector<float>> &t1, float t2) {
	vector<vector<float>> output(t1.size());
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = multiply(t1[i], t2);
	}
	return output;
}

vector<vector<vector<float>>> multiply(vector<vector<vector<float>>> &t1, float t2) {
	vector<vector<vector<float>>> output(t1.size());
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = multiply(t1[i], t2);
	}
	return output;
}

vector<vector<vector<vector<float>>>> multiply(vector<vector<vector<vector<float>>>> &t1, float t2) {
	vector<vector<vector<vector<float>>>> output(t1.size());
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = multiply(t1[i], t2);
	}
	return output;
}

float multiply(vector<float> &t1, vector<float> &t2) { //t1 and t2 have to have the same size
	float output = 0.0;
	for (unsigned i = 0; i < t1.size(); i++) {
		output += t1[i] * t2[i];
	}
	return output;
}

vector<float> multiply(vector<vector<float>> &t1, vector<float> &t2) { //t1[0] and t2 have to have the same size
	vector<float> output(t1.size());
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = multiply(t1[i], t2);
	}
	return output;
}

vector<vector<float>> multiply(vector<vector<float>> &t1, vector<vector<float>> &t2) { //t1[0] and t2 have to have the same size
	vector<vector<float>> output(t1.size(), vector<float>(t2[0].size()));
	vector<vector<float>> t2_t = transpose(t2);
	for (unsigned i = 0; i < t1.size(); i++) {
		for (unsigned j = 0; j < t2[0].size(); j++) {
			output[i][j] = multiply(t1[i], t2_t[j]);
		}
	}
	return output;
}

vector<float> add(vector<float> &t1, vector<float> &t2) {
	vector<float> output(t1.size());
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = t1[i] + t2[i];
	}
	return output;
}

vector<vector<float>> add(vector<vector<float>> &t1, vector<vector<float>> &t2) {
	vector<vector<float>> output(t1.size());
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = add(t1[i], t2[i]);
	}
	return output;
}

vector<vector<vector<float>>> add(vector<vector<vector<float>>> &t1, vector<vector<vector<float>>> &t2) {
	vector<vector<vector<float>>> output(t1.size());
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = add(t1[i], t2[i]);
	}
	return output;
}

vector<vector<vector<vector<float>>>> add(vector<vector<vector<vector<float>>>> &t1, vector<vector<vector<vector<float>>>> &t2) {
	vector<vector<vector<vector<float>>>> output(t1.size());
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = add(t1[i], t2[i]);
	}
	return output;
}

vector<float> add(vector<float> &t1, float t2) {
	vector<float> output(t1.size());
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = t1[i] + t2;
	}
	return output;
}

vector<vector<float>> add(vector<vector<float>> &t1, float t2) {
	vector<vector<float>> output(t1.size());
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = add(t1[i], t2);
	}
	return output;
}

vector<vector<vector<float>>> add(vector<vector<vector<float>>> &t1, float t2) {
	vector<vector<vector<float>>> output(t1.size());
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = add(t1[i], t2);
	}
	return output;
}

vector<vector<vector<vector<float>>>> add(vector<vector<vector<vector<float>>>> &t1, float t2) {
	vector<vector<vector<vector<float>>>> output(t1.size());
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = add(t1[i], t2);
	}
	return output;
}

vector<float> root(vector<float> &t1) {
	vector<float> output(t1.size());
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = sqrt(t1[i]);
	}
	return output;
}

vector<vector<float>> root(vector<vector<float>> &t1) {
	vector<vector<float>> output(t1.size());
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = root(t1[i]);
	}
	return output;
}

vector<vector<vector<float>>> root(vector<vector<vector<float>>> &t1) {
	vector<vector<vector<float>>> output(t1.size());
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = root(t1[i]);
	}
	return output;
}

vector<vector<vector<vector<float>>>> root(vector<vector<vector<vector<float>>>> &t1) {
	vector<vector<vector<vector<float>>>> output(t1.size());
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = root(t1[i]);
	}
	return output;
}

vector<float> divide(vector<float> &t1, vector<float> &t2) {
	vector<float> output(t1.size());
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = t1[i] / t2[i];
	}
	return output;
}

vector<vector<float>> divide(vector<vector<float>> &t1, vector<vector<float>> &t2) {
	vector<vector<float>> output(t1.size());
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = divide(t1[i], t2[i]);
	}
	return output;
}

vector<vector<vector<float>>> divide(vector<vector<vector<float>>> &t1, vector<vector<vector<float>>> &t2) {
	vector<vector<vector<float>>> output(t1.size());
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = divide(t1[i], t2[i]);
	}
	return output;
}

vector<vector<vector<vector<float>>>> divide(vector<vector<vector<vector<float>>>> &t1, vector<vector<vector<vector<float>>>> &t2) {
	vector<vector<vector<vector<float>>>> output(t1.size());
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = divide(t1[i], t2[i]);
	}
	return output;
}

vector<float> pow(vector<float> &t1, float t2) {
	vector<float> output(t1.size());
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = pow(t1[i], t2);
	}
	return output;
}

vector<vector<float>> pow(vector<vector<float>> &t1, float t2) {
	vector<vector<float>> output(t1.size());
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = pow(t1[i], t2);
	}
	return output;
}

vector<vector<vector<float>>> pow(vector<vector<vector<float>>> &t1, float t2) {
	vector<vector<vector<float>>> output(t1.size());
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = pow(t1[i], t2);
	}
	return output;
}

vector<vector<vector<vector<float>>>> pow(vector<vector<vector<vector<float>>>> &t1, float t2) {
	vector<vector<vector<vector<float>>>> output(t1.size());
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = pow(t1[i], t2);
	}
	return output;
}

void addInPlace(vector<float> &t1, vector<float> &t2) {
	for (unsigned i = 0; i < t1.size(); i++) {
		t1[i] += t2[i];
	}
}

void addInPlace(vector<vector<float>> &t1, vector<vector<float>> &t2) {
	for (unsigned i = 0; i < t1.size(); i++) {
		addInPlace(t1[i], t2[i]);
	}
}

void addInPlace(vector<vector<vector<float>>> &t1, vector<vector<vector<float>>> &t2) {
	for (unsigned i = 0; i < t1.size(); i++) {
		addInPlace(t1[i], t2[i]);
	}
}

void addInPlace(vector<vector<vector<vector<float>>>> &t1, vector<vector<vector<vector<float>>>> &t2) {
	for (unsigned i = 0; i < t1.size(); i++) {
		addInPlace(t1[i], t2[i]);
	}
}

class Conv {
public:
	int num_filters;
	int num_of_inputs, input_size1, input_size2;
	int conv_size1, conv_size2;
	int num_windows1, num_windows2;
	vector<vector<vector<float>>> filters;
	vector<float> biases;

	Conv(int f, int c1, int c2, int n, int s1, int s2) {
		num_filters = f;
		conv_size1 = c1;
		conv_size2 = c2;
		num_of_inputs = n;
		input_size1 = s1;
		input_size2 = s2;
		num_windows1 = input_size1 - conv_size1 + 1;
		num_windows2 = input_size2 - conv_size2 + 1;

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

	vector<vector<vector<float>>> forward(vector<vector<vector<float>>> &input) {
		vector<vector<vector<float>>> output(num_of_inputs * num_filters, vector<vector<float>>(num_windows1, vector<float>(num_windows2)));
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
								output[cur_featureMap + num_of_inputs * cur_filter][i][j] += input[cur_featureMap][i + m][j + n] * filters[cur_filter][m][n];
							}
						}
					}
				}
			}
		}

		return output;
	}

	tuple<vector<vector<vector<float>>>, vector<float>, vector<vector<vector<float>>>> backprop(vector<vector<vector<float>>> &loss_gradient,
			vector<vector<vector<float>>> &last_input) {
		vector<vector<vector<float>>> filter_gradient(num_filters, vector<vector<float>>(conv_size1, vector<float>(conv_size2, 0.0)));
		vector<float> bias_gradient(num_filters, 0.0);
		vector<vector<vector<float>>> loss_input(num_of_inputs, vector<vector<float>>(input_size1, vector<float>(input_size2, 0.0)));

		for (int cur_filter = 0; cur_filter < num_filters; cur_filter++) { //per filter
			for (int cur_featureMap = 0; cur_featureMap < num_of_inputs; cur_featureMap++) { //per input
				for (int i = 0; i < num_windows1; i++) {
					//per region
					for (int j = 0; j < num_windows2; j++) {
						// per region
						//matrix multiplication and summation
						for (int m = 0; m < conv_size1; m++) {
							for (int n = 0; n < conv_size2; n++) {
								filter_gradient[cur_filter][m][n] += loss_gradient[cur_featureMap + num_of_inputs * cur_filter][i][j]
										* last_input[cur_featureMap][i + m][j + n];
								loss_input[cur_featureMap][i + m][j + n] += loss_gradient[cur_featureMap + num_of_inputs * cur_filter][i][j]
										* filters[cur_filter][m][n];
							}
						}

						bias_gradient[cur_filter] += loss_gradient[cur_featureMap * num_filters + cur_filter][i][j];
					}
				}
			}
		}

		return {filter_gradient, bias_gradient, loss_input};
	}
};

class MaxPool {
public:
	int num_of_inputs, input_size1, input_size2;
	int window, stride;
	int output_size1, output_size2;

	MaxPool(int w, int s, int n, int s1, int s2) {
		window = w;
		stride = s;
		num_of_inputs = n;
		input_size1 = s1;
		input_size2 = s2;
		output_size1 = (input_size1 - window) / stride + 1;
		output_size2 = (input_size2 - window) / stride + 1;
	}

	vector<vector<vector<float>>> forward(vector<vector<vector<float>>> &input) {
		vector<vector<vector<float>>> output(num_of_inputs, vector<vector<float>>(output_size1, vector<float>(output_size2, 0.0)));
		for (int cur_featureMap = 0; cur_featureMap < num_of_inputs; cur_featureMap++) { //per input
			for (int i = 0; i < input_size1 - window; i += stride) {
				//per region
				for (int j = 0; j < input_size2 - window; j += stride) {
					// per region

					//matrix max pooling
					float max = input[cur_featureMap][i][j];
					for (int m = 0; m < window; m++) {
						for (int n = 0; n < window; n++)
							if (max < input[cur_featureMap][i + m][j + n])
								max = input[cur_featureMap][i + m][j + n];

						output[cur_featureMap][i / stride][j / stride] = max;
					}
				}
			}
		}

		return output;
	}

	vector<vector<vector<float>>> backprop(vector<vector<vector<float>>> &loss_gradient, vector<vector<vector<float>>> &last_input) {
		vector<vector<vector<float>>> loss_input(num_of_inputs, vector<vector<float>>(input_size1, vector<float>(input_size2, 0.0)));
		for (int cur_featureMap = 0; cur_featureMap < num_of_inputs; cur_featureMap++) { //per input
			for (int i = 0; i < input_size1 - window; i += stride) {
				//per region
				for (int j = 0; j < input_size2 - window; j += stride) {
					// per region

					//matrix max pooling
					float max = last_input[cur_featureMap][i][j];
					int indexX = 0;
					int indexY = 0;
					for (int m = 0; m < window; m++) {
						for (int n = 0; n < window; n++) {
							if (max < last_input[cur_featureMap][i + m][j + n]) {
								max = last_input[cur_featureMap][i + m][j + n];
								indexX = m;
								indexY = n;
							}
						}
					}

					//set only the lossInput of the "pixel" max pool kept
					loss_input[cur_featureMap][i + indexX][j + indexY] = loss_gradient[cur_featureMap][i / stride][j / stride];
				}
			}
		}

		return loss_input;
	}
};

class FullyConnectedLayer {
public:
	int num_of_inputs, input_size1, input_size2;
	int num_weights;
	int total_size;
	vector<vector<float>> weights;
	vector<float> biases;

	FullyConnectedLayer(int w, int n, int s1, int s2) {
		num_weights = w;
		num_of_inputs = n;
		input_size1 = s1;
		input_size2 = s2;
		total_size = num_of_inputs * input_size1 * input_size2;
		weights.resize(num_weights, vector<float>(total_size));
		biases.resize(num_weights, 0.0);

		normal_distribution<float> distribution(0.0, 1.0);
		for (int i = 0; i < num_weights; i++) {
			for (int j = 0; j < total_size; j++) {
				random_device dev;
				default_random_engine generator(dev());
				weights[i][j] = distribution(generator) / (total_size);
			}
		}
	}

	vector<float> forward(vector<float> &input) {
		vector<float> output = multiply(weights, input);
		output = add(output, biases);
		return output;
	}

	tuple<vector<vector<float>>, vector<float>, vector<float>> backprop(vector<float> &loss_gradient, vector<float> &last_input) {
		vector<vector<float>> loss_gradient_2D = addDim(loss_gradient);
		vector<vector<float>> last_input_t = transpose(last_input);
		vector<vector<float>> weight_gradient = multiply(loss_gradient_2D, last_input_t);

		vector<float> bias_gradient = copy(loss_gradient);

		vector<vector<float>> weights_t = transpose(weights);
		vector<float> loss_input = multiply(weights_t, loss_gradient);

		return {weight_gradient, bias_gradient, loss_input};
	}
};

class CNN {
public:
	static const int sizeX = 28;
	static const int sizeY = 28;
	static const int num_conv_layers = 1;
	const int num_filters = 8;
	const int pool_layers_window = 2;
	const int pool_layers_stride = 2;
	const int conv_size1 = 3;
	const int conv_size2 = 3;
	const int num_weights = 10;
	const float EPSILON = 1 * 10 ^ (-8);
	int step=0;

	vector<Conv> conv_layers;
	vector<MaxPool> pooling_layers;
	FullyConnectedLayer *connected_layer;

	vector<vector<vector<vector<float>>>> first_momentum_filters;
	vector<vector<vector<vector<float>>>> second_momentum_filters;
	vector<vector<float>> first_momentum_conv_biases;
	vector<vector<float>> second_momentum_conv_biases;
	vector<vector<float>> first_momentum_weights;
	vector<vector<float>> second_momentum_weights;
	vector<float> first_momentum_conn_biases;
	vector<float> second_momentum_conn_biases;

	CNN() {
		int currX = sizeX;
		int currY = sizeY;
		int images = 1;
		for (unsigned i = 0; i < num_conv_layers; i++) {

			vector<vector<vector<float>>> firstMomentum(num_filters, vector<vector<float>>(conv_size1, vector<float>(conv_size2, 0.0)));
			first_momentum_filters.push_back(firstMomentum);
			vector<vector<vector<float>>> secondMomentum(num_filters, vector<vector<float>>(conv_size1, vector<float>(conv_size2, 0.0)));
			second_momentum_filters.push_back(secondMomentum);
			vector<float> firstMomentumBiases(num_filters, 0.0);
			first_momentum_conv_biases.push_back(firstMomentumBiases);
			vector<float> secondMomentumBiases(num_filters, 0.0);
			second_momentum_conv_biases.push_back(secondMomentumBiases);

			conv_layers.push_back(Conv(num_filters, conv_size1, conv_size2, images, currX, currY));
			currX -= (conv_size1 - 1);
			currY -= (conv_size2 - 1);
			images *= num_filters;
			pooling_layers.push_back(MaxPool(pool_layers_window, pool_layers_stride, images, currX, currY));
			currX = (currX - pool_layers_window) / pool_layers_stride + 1;
			currY = (currY - pool_layers_window) / pool_layers_stride + 1;
		}
		connected_layer = new FullyConnectedLayer(num_weights, images, currX, currY);
		first_momentum_weights.resize((num_weights), vector<float>(images * currX * currY, 0.0));
		second_momentum_weights.resize((num_weights), vector<float>(images * currX * currY, 0.0));
		first_momentum_conn_biases.resize(num_weights, 0.0);
		second_momentum_conn_biases.resize(num_weights, 0.0);
	}

	tuple<float, int, tuple<vector<vector<vector<vector<float>>>>, vector<vector<float>>, vector<vector<float>>, vector<float>>> conv(
			vector<vector<vector<float>>> &image, int label) {
		vector<vector<vector<vector<float>>>> z;
		z.push_back(image);
		for (int i = 0; i < num_conv_layers; i++) {
			vector<vector<vector<float>>> help = conv_layers[i].forward(z.back());
			ReLu(help);
			z.push_back(help);
			help = pooling_layers[i].forward(z.back());
			z.push_back(help);
		}
		vector<float> h = flatten(z.back());
		vector<float> res = (*connected_layer).forward(h);

		vector<float> probs = softmax(res);

		float loss = -log(probs[label]);
		int correct = 0;

		int argmax = 0;
		for (int i = 0; i < num_weights; i++)
			if (probs[i] >= probs[argmax])
				argmax = i;

		if (argmax == label)
			correct = 1;

		probs[label] -= 1;

		vector<vector<vector<vector<float>>>> filter_gradients;
		vector<vector<float>> conv_bias_gradients;
		vector<vector<float>> weight_gradient;
		vector<float> conn_bias_gradient;

		tuple<vector<vector<float>>, vector<float>, vector<float>> helpconn = (*connected_layer).backprop(probs, h);
		weight_gradient = get<0>(helpconn);
		conn_bias_gradient = get<1>(helpconn);
		vector<vector<vector<float>>> helpback = deflatten(get<2>(helpconn), (*connected_layer).num_of_inputs, (*connected_layer).input_size1,
				(*connected_layer).input_size2);
		for (int i = num_conv_layers - 1; i > -1; i--) {
			helpback = pooling_layers[i].backprop(helpback, z[2 * i + 1]);
			ReLuPrime(helpback, z[2 * i + 1]);
			tuple<vector<vector<vector<float>>>, vector<float>, vector<vector<vector<float>>>> helpconv = conv_layers[i].backprop(helpback, z[2 * i]);
			filter_gradients.push_back(get<0>(helpconv));
			conv_bias_gradients.push_back(get<1>(helpconv));
			helpback = get<2>(helpconv);

		}

		return {loss, correct, {filter_gradients, conv_bias_gradients, weight_gradient, conn_bias_gradient}};
	}

	tuple<float, float> learn(float alpha, float beta1, float beta2, vector<vector<vector<float>> > &x_batch, vector<int> &y_batch, int batchSize) {
		step++;

		vector<vector<vector<float>> > image(1, vector<vector<float>>(sizeX, vector<float>(sizeY)));
		image[0] = x_batch[0];
		int label = y_batch[0];
		tuple<float, int, tuple<vector<vector<vector<vector<float>>>>, vector<vector<float>>, vector<vector<float>>, vector<float>>> t = conv(image, label);
		tuple<vector<vector<vector<vector<float>>>>, vector<vector<float>>, vector<vector<float>>, vector<float>> thelp = get<2>(t);
		vector<vector<vector<vector<float>>>> filterGradients = get<0>(thelp);
		vector<vector<float>> filterBiases = get<1>(thelp);
		vector<vector<float>> weightGradient = get<2>(thelp);
		vector<float> weightBiases = get<3>(thelp);

		float loss = get<0>(t);
		float correct = get<1>(t);
		for (int i = 1; i < batchSize; i++) {
			image[0] = x_batch[i];
			label = y_batch[i];
			t = conv(image, label);
			loss += get<0>(t);
			correct += get<1>(t);
			thelp = get<2>(t);
			filterGradients = add(filterGradients, get<0>(thelp));
			filterBiases = add(filterBiases, get<1>(thelp));
			weightGradient = add(weightGradient, get<2>(thelp));
			weightBiases = add(weightBiases, get<3>(thelp));
		}

		updateFilterMomentum(filterGradients, filterBiases, beta1, beta2, batchSize); //if you want to use adam uncomment this and comment updateFilters2, updateWeights2
		updateFilters(alpha, beta1, beta2);
		updateWeightMomentum(weightGradient, weightBiases, beta1, beta2, batchSize);
		updateWeights(alpha, beta1, beta2);

		/*updateFilters2(alpha, filterGradients, filterBiases, batchSize);
		 updateWeights2(alpha, weightGradient, weightBiases, batchSize);*/

		return {loss, correct};
	}

	void updateFilterMomentum(vector<vector<vector<vector<float>>>> &filterGradients, vector<vector<float>> &filterBiases, float beta1, float beta2,
			int batchSize) {
		vector<vector<vector<vector<float>>>> first_momentum_filters_adj = multiply(first_momentum_filters, beta1);
		vector<vector<vector<vector<float>>>> filterGradients_adj1 = multiply(filterGradients, (1 - beta1) / batchSize);
		first_momentum_filters = add(first_momentum_filters_adj, filterGradients_adj1);

		vector<vector<float>> first_momentum_conv_biases_adj = multiply(first_momentum_conv_biases, beta1);
		vector<vector<float>> filterBiases_adj1 = multiply(filterBiases, (1 - beta1) / batchSize);
		first_momentum_conv_biases = add(first_momentum_conv_biases_adj, filterBiases_adj1);

		vector<vector<vector<vector<float>>>> second_momentum_filters_adj = multiply(second_momentum_filters, beta2);
		vector<vector<vector<vector<float>>>> filterGradients_adj2_temp1 = multiply(filterGradients, 1.0 / batchSize);
		vector<vector<vector<vector<float>>>> filterGradients_adj2_temp2 = pow(filterGradients_adj2_temp1, 2);
		vector<vector<vector<vector<float>>>> filterGradients_adj2 = multiply(filterGradients_adj2_temp2, (1 - beta2));
		second_momentum_filters = add(filterGradients_adj2, filterGradients_adj2);

		vector<vector<float>> second_momentum_conv_biases_adj = multiply(second_momentum_conv_biases, beta2);
		vector<vector<float>> filterBiases_adj2_temp1 = multiply(filterBiases, 1.0 / batchSize);
		vector<vector<float>> filterBiases_adj2_temp2 = pow(filterBiases_adj2_temp1, 2);
		vector<vector<float>> filterBiases_adj2 = multiply(filterBiases_adj2_temp2, (1 - beta2));
		second_momentum_conv_biases = add(second_momentum_conv_biases_adj, filterBiases_adj2);
	}

	void updateFilters(float alpha, float beta1, float beta2) {
		vector<vector<vector<vector<float>>>> first_momentum_filters_adj = multiply(first_momentum_filters, -alpha/(1-pow(beta1, step)));
		vector<vector<vector<vector<float>>>> second_momentum_filters_adj_temp1 =  multiply(second_momentum_filters, 1/(1-pow(beta2, step)));
		vector<vector<vector<vector<float>>>> second_momentum_filters_adj_temp2 = root(second_momentum_filters_adj_temp1);
		vector<vector<vector<vector<float>>>> second_momentum_filters_adj = add(second_momentum_filters_adj_temp2, EPSILON);
		vector<vector<vector<vector<float>>>> filter_change = divide(first_momentum_filters_adj, second_momentum_filters_adj);
		for (unsigned i = 0; i < first_momentum_filters.size(); i++) {
		 conv_layers[i].filters = add(conv_layers[i].filters, filter_change[i]);
		 }

		vector<vector<float>> first_momentum_conv_biases_adj = multiply(first_momentum_conv_biases, -alpha/(1-pow(beta1, step)));
		vector<vector<float>> second_momentum_conv_biases_adj_temp1=multiply(second_momentum_conv_biases, 1/(1-pow(beta2, step)));
		vector<vector<float>> second_momentum_conv_biases_adj_temp2 = root(second_momentum_conv_biases_adj_temp1);
		vector<vector<float>> second_momentum_conv_biases_adj = add(second_momentum_conv_biases_adj_temp2, EPSILON);
		vector<vector<float>> conv_biases_change = divide(first_momentum_conv_biases_adj, second_momentum_conv_biases_adj);

		for (unsigned i = 0; i < first_momentum_conv_biases.size(); i++) {
		 conv_layers[i].biases = add(conv_layers[i].biases, conv_biases_change[i]);
		 }
	}

	void updateFilters2(float alpha, vector<vector<vector<vector<float>>>> &filterGradients, vector<vector<float>> &filterBiases, int batchSize) {
		vector<vector<vector<vector<float>>>> filterGradients_adj = multiply(filterGradients, -alpha / batchSize);
		for (unsigned i = 0; i < filterGradients.size(); i++) {
			conv_layers[i].filters = add(conv_layers[i].filters, filterGradients_adj[i]);
		}

		vector<vector<float>> filterBiases_adj = multiply(filterBiases, -alpha / batchSize);
		for (unsigned i = 0; i < filterBiases.size(); i++) {
			conv_layers[i].biases = add(conv_layers[i].biases, filterBiases_adj[i]);
		}
	}

	void updateWeightMomentum(vector<vector<float>> &weightGradient, vector<float> &weightBiases, float beta1, float beta2, int batchSize) {
		vector<vector<float>> first_momentum_weights_adj = multiply(first_momentum_weights, beta1);
		vector<vector<float>> weightGradient_adj1 = multiply(weightGradient, (1 - beta1) / batchSize);
		first_momentum_weights = add(first_momentum_weights_adj, weightGradient_adj1);

		vector<float> first_momentum_conn_biases_adj = multiply(first_momentum_conn_biases, beta1);
		vector<float> weightBiases_adj1 = multiply(weightBiases, (1 - beta1) / batchSize);
		first_momentum_conn_biases = add(first_momentum_conn_biases_adj, weightBiases_adj1);

		vector<vector<float>> second_momentum_weights_adj = multiply(second_momentum_weights, beta2);
		vector<vector<float>> weightGradient_adj2_temp1 = multiply(weightGradient, 1.0 / batchSize);
		vector<vector<float>> weightGradient_adj2_temp2 = pow(weightGradient_adj2_temp1, 2);
		vector<vector<float>> weightGradient_adj2 = multiply(weightGradient_adj2_temp2, (1 - beta2));
		second_momentum_weights= add(second_momentum_weights_adj, weightGradient_adj2);

		vector<float> second_momentum_conn_biases_adj = multiply(second_momentum_conn_biases, beta2);
		vector<float> weightBiases_adj2_temp1 = multiply(weightBiases, 1.0 / batchSize);
		vector<float> weightBiases_adj2_temp2 = pow(weightBiases_adj2_temp1, 2);
		vector<float> weightBiases_adj2 = multiply(weightBiases_adj2_temp2, (1 - beta2));
		second_momentum_conn_biases = add(second_momentum_conn_biases_adj, weightBiases_adj2);
	}

	void updateWeights(float alpha, float beta1, float beta2) {
		vector<vector<float>> first_momentum_weights_adj = multiply(first_momentum_weights, -alpha/(1-pow(beta1, step)));
		vector<vector<float>> second_momentum_weights_adj_temp1 = multiply(second_momentum_weights, 1/(1-pow(beta2, step)));
		vector<vector<float>> second_momentum_weights_adj_temp2 = root(second_momentum_weights_adj_temp1);
		vector<vector<float>> second_momentum_weights_adj = add(second_momentum_weights_adj_temp2, EPSILON);
		vector<vector<float>> weights_change = divide(first_momentum_weights_adj, second_momentum_weights_adj);
		(*connected_layer).weights = add((*connected_layer).weights, weights_change);

		vector<float> first_momentum_conn_biases_adj = multiply(first_momentum_conn_biases, -alpha/(1-pow(beta1, step)));
		vector<float> second_momentum_conn_biases_adj_temp1 = multiply(second_momentum_conn_biases, 1/(1-pow(beta2, step)));
		vector<float> second_momentum_conn_biases_adj_temp2 = root(second_momentum_conn_biases_adj_temp1);
		vector<float> second_momentum_conn_biases_adj = add(second_momentum_conn_biases_adj_temp2, EPSILON);
		vector<float> conn_biases_change = divide(first_momentum_conn_biases_adj, second_momentum_conn_biases_adj);
		(*connected_layer).biases = add((*connected_layer).biases, conn_biases_change);
	}

	void updateWeights2(float alpha, vector<vector<float>> &weigthGradient, vector<float> &weightBiases, int batchSize) {
		vector<vector<float>> weigthGradient_adj = multiply(weigthGradient, -alpha / batchSize);
		(*connected_layer).weights = add((*connected_layer).weights, weigthGradient_adj);

		vector<float> weightBiases_adj = multiply(weightBiases, -alpha / batchSize);
		(*connected_layer).biases = add((*connected_layer).biases, weightBiases_adj);
	}

};

void read_trainingData(string filename, vector<vector<float>> &training_images, vector<int> &correct_lables);

int main() {
	try {
		vector<vector<float>> training_images(42000, vector<float>(784));
		vector<int> correct_lables(42000);

		/* Einlesen der Trainingsdaten*/
		read_trainingData("train.txt", training_images, correct_lables);

		/*Vorbereiten des Netzwerks für das Training*/
		const int batchSize = 32;
		const int imageSize = 28;
		const int num_steps = 1000;

		float endLoss;
		float endCorr;

		vector<vector<vector<float>>> batch_images(batchSize, vector<vector<float>>(imageSize, vector<float>(imageSize)));
		vector<int> batch_lables(batchSize);
		CNN cnn; //Das benutzte Netzwerk. Topologieänderungen bitte in der Klasse CNN
		const float alpha = 0.001;		//Lernrate
		const float beta1 = 0.99;		//Erstes Moment
		const float beta2 = 0.999;		//Zweites Moment
		auto training_startTime = chrono::system_clock::now(); // Interner Timer um die Laufzeit zu messen

		for (int i = 0; i < num_steps; i++) {

			/* Vorberiten des Trainingsbatches */
			int randIndex = rand() % (42000 - batchSize);
			for (unsigned j = 0; j < batchSize; j++) { //erstelle einen zufälligen Batch für das Training
				for (int k = 0; k < 784; k++) //Reformatierung des flachen Vektors in Zeilen und Spalten
					batch_images[j][k / imageSize][k % imageSize] = training_images[j + randIndex][k];

				batch_lables[j] = correct_lables[j + randIndex];
			}

			tuple<float, float> res = cnn.learn(alpha, beta1, beta2, batch_images, batch_lables, batchSize);

			float loss = get<0>(res);
			float correct = get<1>(res);

			cout << "Batch " << i + 1 << " Average Loss " << loss / batchSize << " Accuracy " << correct / batchSize << "\n";

			/*zusätzliche Daten zum Topologievergleich*/
			if (num_steps - i <= 10) {
				endLoss += loss;
				endCorr += correct;
			}

			//cout << "Batch " << i + 1 << " Average Loss " << loss / batchSize << " Accuracy " << correct / batchSize << "\n";
		}

		auto training_endTime = chrono::system_clock::now();
		chrono::duration<double> totalTime = training_endTime - training_startTime;
		cout << "Total time: " << (int) (totalTime.count() / 60) << " minutes " << (int) (totalTime.count()) % 60 << " seconds\n";
		cout << "Average batch time: " << (totalTime.count() / num_steps) << "seconds\n";
		cout << "Average loss in last 10 batches:" << endLoss / (10 * batchSize) << ", Average accuracy in last 10 batches: " << endCorr / (10 * batchSize)
				<< "\n";
		return 0;
	} catch (const exception&) {
		return -1;
	}
}

void read_trainingData(string filename, vector<vector<float>> &training_images, vector<int> &correct_lables) {
	ifstream myFile(filename);
	if (myFile.is_open()) {
		int lineNum = 0;
		string line;
		while (getline(myFile, line)) {
			istringstream ss(line);
			string token;
			int i = 0;
			while (getline(ss, token, '\t')) {
				int digit = stoi(token, nullptr);
				if (i == 0)			//erste Zahl jeder Zeile ist das lable
					correct_lables[lineNum] = digit;
				else
					//der Rest das Graustufenbild
					training_images[lineNum][i - 1] = static_cast<float>(digit) / static_cast<float>(255);

				i++;
			}

			lineNum++;
		}
		myFile.close();
	}

}
