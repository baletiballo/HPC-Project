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

using namespace std;

class Conv5x5 {
public:
	int num_filters;
	int size1, size2, size3;
	static const int conv_size = 5;
	vector<vector<vector<float>>> filters;
	vector<float> biases;

	vector<vector<vector<float>>> last_input;

	Conv5x5(int n, int s1, int s2, int s3) {
		num_filters = n;
		size1 = s1;
		size2 = s2;
		size3 = s3;
		filters.resize(conv_size, vector<vector<float>>(conv_size, vector<float>(num_filters)));
		biases.resize(num_filters, 0.0);

		normal_distribution<float> distribution(0.0, 1.0);
		for (int i = 0; i < conv_size; i++) {
			for (int j = 0; j < conv_size; j++) {
				for (int cur_filter = 0; cur_filter < num_filters; cur_filter++) {
					random_device dev;
					default_random_engine generator(dev());
					filters[i][j][cur_filter] = distribution(generator) / (conv_size * conv_size);
				}
			}
		}
	}

	vector<vector<vector<float>>> forward(vector<vector<vector<float>>> &input) {
		const int numWindows = size2 - conv_size + 1;
		vector<vector<vector<float>>> output(size1 * num_filters, vector<vector<float>>(numWindows, vector<float>(size3 - (conv_size - 1), 0.0)));
		for (int i = 0; i < numWindows; i++) {
			//per region
			for (int j = 0; j < numWindows; j++) {
				// per region
				for (int cur_filter = 0; cur_filter < num_filters; cur_filter++) {
					//per filter
					for (int cur_featureMap = 0; cur_featureMap < size1; cur_featureMap++) {
						//per passed representation
						output[cur_featureMap * num_filters + cur_filter][i][j] = biases[cur_filter];

						//set output at i j for the input representation cur_featureMap when filter cur_filter is applied
						//matrix multiplication and summation
						for (int m = 0; m < conv_size; m++)
							for (int n = 0; n < conv_size; n++)
								output[cur_featureMap * num_filters + cur_filter][i][j] += input[cur_featureMap][i + m][j + n] * filters[m][n][cur_filter];
					}
				}
			}
		}

		last_input = input;
		return output;
	}

	tuple<vector<vector<vector<float>>>, vector<float>, vector<vector<vector<float>>>> backprop(vector<vector<vector<float>>> &lossGradient) {
		vector<vector<vector<float>>> filterGradient(conv_size, vector<vector<float>>(conv_size, vector<float>(num_filters, 0.0)));
		vector<float> filterBias(num_filters, 0.0);
		vector<vector<vector<float>>> lossInput(size1, vector<vector<float>>(size2, vector<float>(size3, 0.0)));

		for (int i = 0; i < size2 - (conv_size - 1); i++) {
			//per region
			for (int j = 0; j < size3 - (conv_size - 1); j++) {
				// per region
				for (int cur_filter = 0; cur_filter < num_filters; cur_filter++) {
					//per filter
					for (int cur_featureMap = 0; cur_featureMap < size1; cur_featureMap++) {
						//per passed representation
						//matrix multiplication and summation
						for (int m = 0; m < conv_size; m++) {
							for (int n = 0; n < conv_size; n++) {
								filterGradient[m][n][cur_filter] += lossGradient[cur_featureMap * num_filters + cur_filter][i][j]
										* last_input[cur_featureMap][i + m][j + n];
								lossInput[cur_featureMap][i + m][j + n] += lossGradient[cur_featureMap * num_filters + cur_filter][i][j]
										* filters[m][n][cur_filter];
							}
						}

						filterBias[cur_filter] += lossGradient[cur_featureMap * num_filters + cur_filter][i][j];
					}
				}
			}
		}

		return {filterGradient, filterBias, lossInput};
	}
};

class MaxPool {
public:
	int size1, size2, size3;
	int window, stride;

	vector<vector<vector<float>>> last_input;

	MaxPool(int w, int s, int s1, int s2, int s3) {
		window = w;
		stride = s;
		size1 = s1;
		size2 = s2;
		size3 = s3;
	}

	vector<vector<vector<float>>> forward(vector<vector<vector<float>>> &input) {
		vector<vector<vector<float>> > output(size1, vector<vector<float>>((size2 - window) / stride + 1, vector<float>((size3 - window) / stride + 1, 0.0)));
		for (int i = 0; i < size2 - window; i += stride) {
			//per region
			for (int j = 0; j < size3 - window; j += stride) {
				// per region
				for (int cur_featureMap = 0; cur_featureMap < size1; cur_featureMap++) {
					//per passed representation
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

		last_input = input;
		return output;
	}

	vector<vector<vector<float>>> backprop(vector<vector<vector<float>>> &lossGradient) {
		vector<vector<vector<float>>> lossInput(size1, vector<vector<float>>(size2, vector<float>(size3, 0.0)));

		for (int i = 0; i < size2 - window; i += stride) {
			//per region
			for (int j = 0; j < size3 - window; j += stride) {
				// per region
				for (int cur_featureMap = 0; cur_featureMap < size1; cur_featureMap++) {
					//per passed representation
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
					lossInput[cur_featureMap][i + indexX][j + indexY] = lossGradient[cur_featureMap][i / stride][j / stride];
				}
			}
		}

		return lossInput;
	}
};

class FullyConnectedLayer {
public:
	int num_featureMaps; //Number of feature maps the convolutional Layers generate
	int size2, size3; //Dimensions of the feature maps
	static const int num_weights = 10; //Number of 
	vector<vector<float>> weights;
	vector<float> biases;

	vector<float> last_inputVector;
	vector<float> last_totals;
	float last_sum = 0.0;

	FullyConnectedLayer(int s1, int s2, int s3) {
		num_featureMaps = s1;
		size2 = s2;
		size3 = s3;
		weights.resize(num_featureMaps * size2 * size3, vector<float>(num_weights));
		biases.resize(num_weights, 0.0);

		normal_distribution<float> distribution(0.0, 1.0);
		for (int i = 0; i < num_featureMaps * size2 * size3; i++) {
			for (int j = 0; j < num_weights; j++) {
				random_device dev;
				default_random_engine generator(dev());
				weights[i][j] = distribution(generator) / (num_featureMaps * size2 * size3);
			}
		}

		last_totals.resize(num_weights);
	}

	vector<float> forward(vector<vector<vector<float>>> &input) {
		vector<float> output(num_weights);
		for (int i = 0; i < num_weights; i++)
			output[i] = biases[i];

		//flatten (the curve xD)
		vector<float> inputVector(num_featureMaps * size2 * size3);
		for (int i = 0; i < num_featureMaps; i++)
			for (int j = 0; j < size2; j++)
				for (int k = 0; k < size3; k++)
					inputVector[i * size2 * size3 + j * size3 + k] = input[i][j][k];

		for (int i = 0; i < num_featureMaps * size2 * size3; i++) //per feature
			for (int j = 0; j < num_weights; j++) //per weights
				output[j] += inputVector[i] * weights[i][j];

		last_inputVector = inputVector;

		//activation function
		float total = 0.0;
		for (int i = 0; i < num_weights; i++) {
			output[i] = exp(output[i]);
			last_totals[i] = output[i];
			total += output[i];
		}
		last_sum = total;

		//normalize
		for (int i = 0; i < num_weights; i++) {
			output[i] = output[i] / total;
		}
		return output;
	}

	tuple<vector<vector<float>>, vector<float>, vector<vector<vector<float>>>> backprop(vector<float> lossGradient) {
		vector<vector<vector<float>>> lossInput(num_featureMaps, vector<vector<float>>(size2, vector<float>(size3, 0.0)));
		vector<vector<float>> weightGradient(num_featureMaps * size2 * size3, vector<float>(num_weights, 0.0));
		vector<float> biasGradient(num_weights, 0.0);

		for (int i = 0; i < num_featureMaps * size2 * size3; i++) {
			for (int j = 0; j < num_weights; j++) {
				weightGradient[i][j] = lossGradient[j] * last_inputVector[i];
			}
		}
		for (int i = 0; i < num_weights; i++) {
			biasGradient[i] = lossGradient[i];
		}
		for (int i = 0; i < num_featureMaps * size2 * size3; i++) {
			for (int j = 0; j < num_weights; j++) {
				lossInput[i / (size2 * size3)][(i / size3) % size2][i % size3] += lossGradient[j] * weights[i][j];
			}
		}

		/*int index = -1;
		 for (int i = 0; i < num_weights; i++) {
		 if (lossGradient[i] != 0)
		 index = i;
		 }

		 const float gradient = lossGradient[index];

		 vector<float> dOutDt(num_weights);
		 for (int i = 0; i < num_weights; i++)
		 dOutDt[i] = -last_totals[index] * last_totals[i] / (last_sum * last_sum);

		 dOutDt[index] = last_totals[index] * (last_sum - last_totals[index]) / (last_sum * last_sum);

		 vector<float> dLdt(num_weights);
		 for (int i = 0; i < num_weights; i++) {
		 dLdt[i] = gradient * dOutDt[i];
		 }

		 for (int i = 0; i < num_featureMaps * size2 * size3; i++) {
		 for (int j = 0; j < num_weights; j++) {
		 lossInput[i / (size2 * size3)][i / size3 % size2][i % size3] += weights[i][j] * dLdt[j];
		 weightGradient[i][j] += last_inputVector[i] * dLdt[j];
		 }
		 }*/

		return {weightGradient, biasGradient, lossInput};
	}
};

class CNN {
public:
	static const int sizeX = 28;
	static const int sizeY = 28;
	static const int num_conv_layers = 1;
	const int conv_layers_num_filters = 8;
	const int pool_layers_window = 2;
	const int pool_layers_stride = 2;
	const float EPSILON = 1 * 10 ^ (-7);

	vector<Conv5x5> conv_layers;
	vector<MaxPool> pooling_layers;
	FullyConnectedLayer *connected_layer;

	vector<vector<vector<vector<float>>>> firstMomentumFilterGradients;
	vector<vector<vector<vector<float>>>> secondMomentumFilterGradients;
	vector<vector<float>> firstMomentumFilterBiases;
	vector<vector<float>> secondMomentumFilterBiases;
	vector<vector<float>> firstMomentumWeightGradient;
	vector<vector<float>> secondMomentumWeightGradient;
	vector<float> firstMomentumWeightBiases;
	vector<float> secondMomentumWeightBiases;

	CNN() {
		int currX = sizeX;
		int currY = sizeY;
		int images = 1;
		for (unsigned i = 0; i < num_conv_layers; i++) {
			connected_layer = new FullyConnectedLayer(images, currX, currY);

			vector<vector<vector<float>>> firstMomentum(Conv5x5::conv_size,
					vector<vector<float>>(Conv5x5::conv_size, vector<float>(conv_layers_num_filters, 0.0)));
			firstMomentumFilterGradients.push_back(firstMomentum);
			vector<vector<vector<float>>> secondMomentum(Conv5x5::conv_size,
					vector<vector<float>>(Conv5x5::conv_size, vector<float>(conv_layers_num_filters, 0.0)));
			secondMomentumFilterGradients.push_back(firstMomentum);
			vector<float> firstMomentumBiases(conv_layers_num_filters, 0.0);
			firstMomentumFilterBiases.push_back(firstMomentumBiases);
			vector<float> secondMomentumBiases(conv_layers_num_filters, 0.0);
			secondMomentumFilterBiases.push_back(secondMomentumBiases);

			conv_layers.push_back(Conv5x5(conv_layers_num_filters, images, currX, currY));
			currX -= 4;
			currY -= 4;
			images *= conv_layers_num_filters;
			pooling_layers.push_back(MaxPool(pool_layers_window, pool_layers_stride, images, currX, currY));
			currX = (currX - pool_layers_window) / pool_layers_stride + 1;
			currY = (currY - pool_layers_window) / pool_layers_stride + 1;
		}
		connected_layer = new FullyConnectedLayer(images, currX, currY);
		firstMomentumWeightGradient.resize((images * currX * currY), vector<float>(FullyConnectedLayer::num_weights));
		secondMomentumWeightGradient.resize((images * currX * currY), vector<float>(FullyConnectedLayer::num_weights));
		firstMomentumWeightBiases.resize(FullyConnectedLayer::num_weights);
		secondMomentumWeightBiases.resize(FullyConnectedLayer::num_weights);
	}

	vector<float> forward(vector<vector<vector<float>> > &image) {
		vector<vector<vector<float>> > help = conv_layers[0].forward(image);
		//ReLu3D(help);
		help = pooling_layers[0].forward(help);
		for (int i = 1; i < num_conv_layers; i++) {
			help = conv_layers[i].forward(help);
			//ReLu3D(help);
			help = pooling_layers[i].forward(help);
		}
		return (*connected_layer).forward(help);
	}

	tuple<vector<vector<vector<vector<float>>>>, vector<vector<float>>, vector<vector<float>>, vector<float>> backprop(vector<float> &res) {
		vector<vector<vector<vector<float>>>> filterGradients;
		vector<vector<float>> filterBiases;
		vector<vector<float>> weigthGradient;
		vector<float> weightBiases;

		tuple<vector<vector<float>>, vector<float>, vector<vector<vector<float>>>> helpconn = (*connected_layer).backprop(res);
		weigthGradient = get<0>(helpconn);
		weightBiases = get<1>(helpconn);
		//ReLu3D(get<2>(helpconn));

		tuple<vector<vector<vector<float>>>, vector<float>, vector<vector<vector<float>>>> helpconv;
		vector<vector<vector<float>>> t = pooling_layers[num_conv_layers - 1].backprop(get<2>(helpconn));
		helpconv = conv_layers[num_conv_layers - 1].backprop(t);
		//ReLu3D(get<2>(helpconv));
		filterGradients.push_back(get<0>(helpconv));
		filterBiases.push_back(get<1>(helpconv));

		for (int i = num_conv_layers - 2; i > -1; i--) {
			t = pooling_layers[i].backprop(get<2>(helpconv));
			helpconv = conv_layers[i].backprop(t);
			//ReLu3D(get<2>(helpconv));
			filterGradients.push_back(get<0>(helpconv));
			filterBiases.push_back(get<1>(helpconv));
		}
		return {filterGradients, filterBiases, weigthGradient, weightBiases};
	}

	tuple<float, int, tuple<vector<vector<vector<vector<float>>>>, vector<vector<float>>, vector<vector<float>>, vector<float>>> conv(
			vector<vector<vector<float>> > &image, int label) {
		vector<float> res = forward(image);
		/*cout << res[label] << "\n";
		 if (res[label] == 1) {
		 cout << "hello " << label << "\n";
		 }*/
		float loss = -log(res[label]);
		int correct = 0;

		int argmax = 0;
		for (int i = 0; i < FullyConnectedLayer::num_weights; i++)
			if (res[i] >= res[argmax])
				argmax = i;

		if (argmax == label)
			correct = 1;

		res[label] -= 1;
		/*for (int i = 0; i < FullyConnectedLayer::num_weights; i++)
		 if (i == label)
		 res[i] = -1 / res[i];
		 else
		 res[i] = 0;*/

		return {loss, correct, backprop(res)};
	}

	tuple<float, float> learn(float alpha, float beta1, float beta2, vector<vector<vector<float>> > &x_batch, vector<int> &y_batch, int batchSize) {
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
			//cout << "loss" << loss << "correct" << correct << "\n";
			thelp = get<2>(t);
			add4DMatrix(filterGradients, get<0>(thelp));
			add2DMatrix(filterBiases, get<1>(thelp));
			add2DMatrix(weightGradient, get<2>(thelp));
			addVectors(weightBiases, get<3>(thelp));
			/*if(i%10==0) {
			 for (unsigned i = 0; i < filterGradients.size(); i++) {
			 for (unsigned j = 0; j < filterGradients.at(i).size(); j++) {
			 for (unsigned k = 0; k < filterGradients.at(i).at(j).size(); k++) {
			 for (unsigned l = 0; l < filterGradients.at(i).at(j).at(k).size(); l++) {
			 cout<<filterGradients.at(i).at(j).at(k).at(l)<<" ";
			 }
			 cout<<"\n";
			 }
			 }
			 }
			 cout<<"\n";
			 }*/
		}

		/*updateFilterMomentum(filterGradients, filterBiases, beta1, beta2, batchSize);
		updateFilters(alpha);
		updateWeightMomentum(weightGradient, weightBiases, beta1, beta2, batchSize);
		updateWeights(alpha);*/

		updateFilters2(alpha, filterGradients, filterBiases, batchSize);
		 updateWeights2(alpha, weightGradient, weightBiases, batchSize);

		return {loss, correct};
	}

	void ReLu1D(vector<float> &t1) {
		for (unsigned i = 0; i < t1.size(); i++) {
			if (t1[i] <= 0) {
				t1[i] = 0;
			}
		}
	}

	void ReLu2D(vector<vector<float>> &t1) {
		for (unsigned i = 0; i < t1.size(); i++) {
			ReLu1D(t1[i]);
		}
	}

	void ReLu3D(vector<vector<vector<float>>> &t1) {
		for (unsigned i = 0; i < t1.size(); i++) {
			ReLu2D(t1[i]);
		}
	}

	void ReLu4D(vector<vector<vector<vector<float>>>> &t1) {
		for (unsigned i = 0; i < t1.size(); i++) {
			ReLu3D(t1[i]);
		}
	}

	void updateFilterMomentum(vector<vector<vector<vector<float>>>> &filterGradients, vector<vector<float>> &filterBiases, float beta1, float beta2,
			int batchSize) {
		for (unsigned i = 0; i < firstMomentumFilterGradients.size(); i++) {
			for (unsigned j = 0; j < firstMomentumFilterGradients.at(i).size(); j++) {
				for (unsigned k = 0; k < firstMomentumFilterGradients.at(i).at(j).size(); k++) {
					for (unsigned l = 0; l < firstMomentumFilterGradients.at(i).at(j).at(k).size(); l++) {
						//cout << firstMomentumFilterGradients.at(i).at(j).at(k).at(l) << " ";
						firstMomentumFilterGradients.at(i).at(j).at(k).at(l) = (beta1 * firstMomentumFilterGradients.at(i).at(j).at(k).at(l))
								+ ((1 - beta1) * filterGradients.at(i).at(j).at(k).at(l)) / batchSize;
					}
					//cout << "\n";
				}
			}
		}
		for (unsigned i = 0; i < firstMomentumFilterBiases.size(); i++) {
			for (unsigned j = 0; j < firstMomentumFilterBiases.at(i).size(); j++) {
				firstMomentumFilterBiases.at(i).at(j) = (beta1 * firstMomentumFilterBiases.at(i).at(j)) + ((1 - beta1) * filterBiases.at(i).at(j)) / batchSize;
			}
		}

		for (unsigned i = 0; i < secondMomentumFilterGradients.size(); i++) {
			for (unsigned j = 0; j < secondMomentumFilterGradients.at(i).size(); j++) {
				for (unsigned k = 0; k < secondMomentumFilterGradients.at(i).at(j).size(); k++) {
					for (unsigned l = 0; l < secondMomentumFilterGradients.at(i).at(j).at(k).size(); l++) {
						secondMomentumFilterGradients.at(i).at(j).at(k).at(l) = (beta2 * secondMomentumFilterGradients.at(i).at(j).at(k).at(l))
								+ ((1 - beta2) * pow((filterGradients.at(i).at(j).at(k).at(l)) / batchSize, 2));
					}
				}
			}
		}
		for (unsigned i = 0; i < secondMomentumFilterBiases.size(); i++) {
			for (unsigned j = 0; j < secondMomentumFilterBiases.at(i).size(); j++) {
				secondMomentumFilterBiases.at(i).at(j) = (beta2 * secondMomentumFilterBiases.at(i).at(j))
						+ ((1 - beta2) * pow((filterBiases.at(i).at(j)) / batchSize, 2));
			}
		}
	}

	void updateFilters(float alpha) {
		for (unsigned i = 0; i < firstMomentumFilterGradients.size(); i++) {
			for (unsigned j = 0; j < firstMomentumFilterGradients.at(i).size(); j++) {
				for (unsigned k = 0; k < firstMomentumFilterGradients.at(i).at(j).size(); k++) {
					for (unsigned l = 0; l < firstMomentumFilterGradients.at(i).at(j).at(k).size(); l++) {
						conv_layers[i].filters[j][k][l] = conv_layers[i].filters[j][k][l]
								- alpha * (firstMomentumFilterGradients.at(i).at(j).at(k).at(l))
										/ (sqrt(secondMomentumFilterGradients.at(i).at(j).at(k).at(l)) + EPSILON);
					}
				}
			}
		}
		for (unsigned i = 0; i < firstMomentumFilterBiases.size(); i++) {
			for (unsigned j = 0; j < firstMomentumFilterBiases.at(i).size(); j++) {
				conv_layers[i].biases[j] = conv_layers[i].biases[j]
						- alpha * (firstMomentumFilterBiases.at(i).at(j)) / (sqrt(secondMomentumFilterBiases.at(i).at(j)) + EPSILON);
			}
		}
	}

	void updateFilters2(float alpha, vector<vector<vector<vector<float>>>> &filterGradients, vector<vector<float>> &filterBiases, int batchSize) {
		for (unsigned i = 0; i < filterGradients.size(); i++) {
			for (unsigned j = 0; j < filterGradients.at(i).size(); j++) {
				for (unsigned k = 0; k < filterGradients.at(i).at(j).size(); k++) {
					for (unsigned l = 0; l < filterGradients.at(i).at(j).at(k).size(); l++) {
						conv_layers[i].filters[j][k][l] = conv_layers[i].filters[j][k][l] - alpha * (filterGradients.at(i).at(j).at(k).at(l)) / batchSize;
					}
				}
			}
		}
		for (unsigned i = 0; i < filterBiases.size(); i++) {
			for (unsigned j = 0; j < filterBiases.at(i).size(); j++) {
				conv_layers[i].biases[j] = conv_layers[i].biases[j] - alpha * (filterBiases.at(i).at(j)) / batchSize;
			}
		}
	}

	void updateWeightMomentum(vector<vector<float>> &weightGradient, vector<float> &weightBiases, float beta1, float beta2, int batchSize) {
		for (unsigned i = 0; i < firstMomentumWeightGradient.size(); i++) {
			for (unsigned j = 0; j < firstMomentumWeightGradient.at(i).size(); j++) {
				firstMomentumWeightGradient.at(i).at(j) = (beta1 * firstMomentumWeightGradient.at(i).at(j))
						+ ((1 - beta1) * weightGradient.at(i).at(j)) / batchSize;
			}
		}
		for (unsigned i = 0; i < firstMomentumWeightBiases.size(); i++) {
			firstMomentumWeightBiases.at(i) = (beta1 * firstMomentumWeightBiases.at(i)) + ((1 - beta1) * weightBiases.at(i)) / batchSize;
		}

		for (unsigned i = 0; i < secondMomentumWeightGradient.size(); i++) {
			for (unsigned j = 0; j < secondMomentumWeightGradient.at(i).size(); j++) {
				secondMomentumWeightGradient.at(i).at(j) = (beta2 * secondMomentumWeightGradient.at(i).at(j))
						+ ((1 - beta2) * pow(weightGradient.at(i).at(j) / batchSize, 2));
			}
		}
		for (unsigned i = 0; i < secondMomentumWeightBiases.size(); i++) {
			secondMomentumWeightBiases.at(i) = (beta2 * secondMomentumWeightBiases.at(i)) + ((1 - beta2) * pow(weightBiases.at(i) / batchSize, 2));
		}
	}

	void updateWeights(float alpha) {
		for (unsigned i = 0; i < firstMomentumWeightGradient.size(); i++) {
			for (unsigned j = 0; j < firstMomentumWeightGradient.at(i).size(); j++) {
				(*connected_layer).weights[i][j] = (*connected_layer).weights[i][j]
						- alpha * (firstMomentumWeightGradient.at(i).at(j)) / (sqrt(secondMomentumWeightGradient.at(i).at(j)) + EPSILON);
			}
		}
		for (unsigned i = 0; i < firstMomentumWeightBiases.size(); i++) {
			(*connected_layer).biases[i] = (*connected_layer).biases[i]
					- alpha * (firstMomentumWeightBiases.at(i)) / (sqrt(secondMomentumWeightBiases.at(i)) + EPSILON);
		}
	}

	void updateWeights2(float alpha, vector<vector<float>> &weigthGradient, vector<float> &weightBiases, int batchSize) {
		for (unsigned i = 0; i < weigthGradient.size(); i++) {
			for (unsigned j = 0; j < weigthGradient.at(i).size(); j++) {
				(*connected_layer).weights[i][j] = (*connected_layer).weights[i][j] - alpha * (weigthGradient.at(i).at(j)) / batchSize;
			}
		}
		for (unsigned i = 0; i < weightBiases.size(); i++) {
			(*connected_layer).biases[i] = (*connected_layer).biases[i] - alpha * (weightBiases.at(i)) / batchSize;
		}
	}

	void addVectors(vector<float> &t1, vector<float> &t2) {
		for (unsigned i = 0; i < t1.size(); i++) {
			t1[i] += t2[i];
		}
	}

	void add2DMatrix(vector<vector<float>> &t1, vector<vector<float>> &t2) {
		for (unsigned i = 0; i < t1.size(); i++) {
			addVectors(t1[i], t2[i]);
		}
	}

	void add3DMatrix(vector<vector<vector<float>>> &t1, vector<vector<vector<float>>> &t2) {
		for (unsigned i = 0; i < t1.size(); i++) {
			add2DMatrix(t1[i], t2[i]);
		}
	}

	void add4DMatrix(vector<vector<vector<vector<float>>>> &t1, vector<vector<vector<vector<float>>>> &t2) {
		for (unsigned i = 0; i < t1.size(); i++) {
			add3DMatrix(t1[i], t2[i]);
		}
	}

};

int main() {
	try {
		vector<vector<float>> x(42000, vector<float>(784));
		vector<int> y(42000);
		//auto line_v = new vector<string>(785);

		//cout << 1 << "\n";

		ifstream myFile("train.txt");
		if (myFile.is_open()) {
			int lineNum = 0;
			string line;
			while (getline(myFile, line)) {
				istringstream ss(line);
				string token;
				int i = 0;
				while (getline(ss, token, '\t')) {
					int digit = stoi(token, nullptr);
					if (i == 0)
						y[lineNum] = digit;
					else
						x[lineNum][i - 1] = static_cast<float>(digit) / static_cast<float>(255);

					i++;
				}

				lineNum++;
			}
			myFile.close();
		}

		//cout << 2 << "\n";

		const int batchSize = 1000;
		const int imageSize = 28;

		vector<vector<vector<float>>> x_batch(batchSize, vector<vector<float>>(imageSize, vector<float>(imageSize)));
		vector<int> y_batch(batchSize);

		/*Conv5x5 conv(filters, 1, imageSize, imageSize);
		 MaxPool pool(poolDimensions, poolDimensions, filters, imageSize - 2, imageSize - 2);
		 FullyConnectedLayer conn(filters, (imageSize - 2) / 2, (imageSize - 2) / 2);*/
		CNN cnn;
		const float alpha = 0.01;		//Lernrate
		const float beta1 = 0.95;	//Erstes Moment
		const float beta2 = 0.99;	//Zweites Moment

		//cout << 3 << "\n";

		for (int i = 0; i < 1000; i++) {
			int randIndex = rand() % (42000 - batchSize);
			for (unsigned j = 0; j < batchSize; j++) {
				for (int k = 0; k < 784; k++)
					x_batch[j][k / imageSize][k % imageSize] = x[j + randIndex][k];

				y_batch[j] = y[j + randIndex];
			}

			tuple<float, float> res = cnn.learn(alpha, beta1, beta2, x_batch, y_batch, batchSize);

			float loss = get<0>(res);
			float correct = get<1>(res);

			cout << "Batch " << i + 1 << " Average Loss " << loss / batchSize << " Accuracy " << correct / batchSize << "\n";
		}

		return 0;
	} catch (const exception&) {
		return -1;
	}
}
