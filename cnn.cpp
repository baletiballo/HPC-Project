#include <vector>
#include <random>
#include "ParallelStuff.h"
#include "Conv.h"
#include "FullyConnectedLayer.h"
#include "MaxPool.h"
#include "Hilfsfunktionen.h"
#include "ReLu.h"

using namespace std;

class CNN {
	static const int sizeX = 28; //Anzahl Pixel in x-Richtung
	static const int sizeY = 28; //Anzahl Pixel in y-Richtung
	static const int num_conv_layers = 1; //Anzahl der Convolutional Layer
	const int num_filters = 8; //Anzahl der Convolutionen pro Conv-Layer
	const int pool_layers_window = 2;
	const int pool_layers_stride = 2;
	const int conv_size1 = 3;
	const int conv_size2 = 3;
	const int num_weights = 10; //Anzahl an Klassifikations Klassen (10, da zehn Ziffern)
	const float EPSILON = 1.0f * pow(10.0f, -8);

	float alpha; //Lernrate
	float beta1; //Erstes Moment
	float beta2; //Zweites Moment
	int batchSize; //Größe der Batches
	int step; //Anzahl der bisher gelernten Batches

	vector<Conv> conv_layers; //Vektor aller Convolutional Layers
	vector<MaxPool> pooling_layers; //Vektor aller Pooling Layers
	FullyConnectedLayer *connected_layer; //Das eine Fully Connected layer

	vector<vector<vector<vector<float>>>> first_momentum_filters;
	vector<vector<vector<vector<float>>>> second_momentum_filters;
	vector<vector<float>> first_momentum_conv_biases;
	vector<vector<float>> second_momentum_conv_biases;
	vector<vector<float>> first_momentum_weights;
	vector<vector<float>> second_momentum_weights;
	vector<float> first_momentum_conn_biases;
	vector<float> second_momentum_conn_biases;

public:
	CNN(float alpha, float beta1,float beta2, int batchSize) : alpha(alpha), beta1(beta1), beta2(beta2), batchSize(batchSize)  {
		step = 1;
		
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

	/**
	 * Jagt @Param image durch das Netzwerk und vergleicht das Ergebnis mit @Param lable
	 *
	 * @param image
	 * @param label
	 * @return Das Tupel (loss, correct, filter_gradients, conv_bias_gradients, weight_gradient, conn_bias_gradient) 
	*/
	tuple<float, bool, vector<vector<vector<vector<float>>>>, vector<vector<float>>, vector<vector<float>>, vector<float>> forward (vector<vector<vector<float>>> &image, int_fast8_t label) {

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

		softmax(res, res);
		float loss = -log(res[label]);
		
		int argmax = 0;
		for (int i = 0; i < num_weights; i++) 
			if (res[i] >= res[argmax]) 
				argmax = i;
		
		bool correct = label == argmax;
		res[label] -= 1;

		vector<vector<vector<vector<float>>>> filter_gradients;
		vector<vector<float>> conv_bias_gradients;
		vector<vector<float>> weight_gradient;
		vector<float> conn_bias_gradient;

		tuple<vector<vector<float>>, vector<float>, vector<float>> helpconn = (*connected_layer).backprop(res, h);
		weight_gradient = get<0>(helpconn);
		conn_bias_gradient = get<1>(helpconn);
		vector<vector<vector<float>>> helpback = deflatten(get<2>(helpconn), (*connected_layer).num_of_inputs, (*connected_layer).input_size1, (*connected_layer).input_size2);
		for (int i = num_conv_layers - 1; i > -1; i--) {
			helpback = pooling_layers[i].backprop(helpback, z[2 * i + 1]);
			ReLuPrime(helpback, z[2 * i + 1]);
			tuple<vector<vector<vector<float>>>, vector<float>, vector<vector<vector<float>>>> helpconv = conv_layers[i].backprop(helpback, z[2 * i]);
			filter_gradients.push_back(get<0>(helpconv));
			conv_bias_gradients.push_back(get<1>(helpconv));
			helpback = get<2>(helpconv);
		}

		return {loss, correct, filter_gradients, conv_bias_gradients, weight_gradient, conn_bias_gradient};
	}

	/**
	 * Description
	 *
	 * @param x_batch
	 * @param y_batch
	 * @return 
	*/
	tuple<float, int_fast8_t> learn	(vector<vector<vector<float>> > &x_batch, vector<int_fast8_t> &y_batch) {
		
		//Das Aktuell zu verarbeitende Bild. Als einelementiger Vektor, da Conv-Layer Vektoren von Bildern nehmen
		vector<vector<vector<float>>> image(1, vector<vector<float>>(sizeX, vector<float>(sizeY)));
		image[0] = x_batch[0];
		int label = y_batch[0]; //Das Lable des aktuellen Bildes
		//Ergebnis des Netzwerks nach dem Forward von image
		tuple<float, bool, vector<vector<vector<vector<float>>>>, vector<vector<float>>, vector<vector<float>>, vector<float>> Result = forward(image,	label);
		vector<vector<vector<vector<float>>>> filterGradients = get<2>(Result);
		vector<vector<float>> filterBiases = get<3>(Result);
		vector<vector<float>> weightGradient = get<4>(Result);
		vector<float> weightBiases = get<5>(Result);

		float loss = get<0>(Result);
		int_fast8_t correct = get<1>(Result);

		for (int i = 1; i < batchSize; i++) {
			image[0] = x_batch[i];
			label = y_batch[i];

			Result = forward(image, label);
			loss += get<0>(Result);
			correct += get<1>(Result);

			addGradients(get<2>(Result), get<3>(Result), get<4>(Result), get<5>(Result), filterGradients, filterBiases, weightGradient,	weightBiases);
		}
		
		//ADAM learning
		updateFilters(filterGradients, filterBiases);
		updateWeights(weightGradient, weightBiases);
		step++;

		return {loss, correct};
	}

	/**
	 * Description
	 *
	 * @param t1
	 * @param t2
	 * @param t3
	 * @param t4
	 * @param o1
	 * @param o2
	 * @param o3
	 * @param o4
	 * @return
	*/
	void addGradients(vector<vector<vector<vector<float>>>> &t1, vector<vector<float>> &t2, vector<vector<float>> &t3, vector<float> &t4, vector<vector<vector<vector<float>>>> &o1, vector<vector<float>> &o2, vector<vector<float>> &o3, vector<float> &o4) {
		for (unsigned i = 0; i < t1.size(); i++) {
			for (unsigned j = 0; j < t1[i].size(); j++) {
				for (unsigned k = 0; k < t1[i][j].size(); k++) 
					for (unsigned l = 0; l < t1[i][j][k].size(); l++) 
						o1[i][j][k][l] += t1[i][j][k][l];

				o2[i][j] += t2[i][j];
			}
		}

		for (unsigned i = 0; i < t3.size(); i++) {
			for (unsigned j = 0; j < t3[i].size(); j++) 
				o3[i][j] += t3[i][j];

			o4[i] += t4[i];
		}
	}

	/**
	 * Description
	 *
	 * @param filterGradients
	 * @param filterBiases
	 * @return
	*/
	void updateFilters(vector<vector<vector<vector<float>>>> &filterGradients, vector<vector<float>> &filterBiases) {
		float corr1 = 1.0f;// - pow(beta1, step); //Korrekturterm des erstes Moments
		float corr2 = 1.0f;// - pow(beta2, step); //Korrekturterm des zweiten Moments

		for (unsigned i = 0; i < first_momentum_filters.size(); i++) {
			for (unsigned j = 0; j < first_momentum_filters[i].size(); j++) {
				for (unsigned k = 0; k < first_momentum_filters[i][j].size(); k++) {
					for (unsigned l = 0; l < first_momentum_filters[i][j][k].size(); l++) {
						first_momentum_filters[i][j][k][l] = beta1 * first_momentum_filters[i][j][k][l] + (1.0 - beta1) * filterGradients[i][j][k][l] / batchSize;
						second_momentum_filters[i][j][k][l] = beta1 * second_momentum_filters[i][j][k][l] + (1.0 - beta1) * pow(filterGradients[i][j][k][l] / batchSize, 2);

						conv_layers[i].filters[j][k][l] = conv_layers[i].filters[j][k][l] - alpha * ((first_momentum_filters[i][j][k][l] / corr1) / (sqrt(second_momentum_filters[i][j][k][l] / corr2) + EPSILON));
					}
				}
				first_momentum_conv_biases[i][j] = beta1 * first_momentum_conv_biases[i][j] + (1.0 - beta1) * filterBiases[i][j] / batchSize;
				second_momentum_conv_biases[i][j] = beta1 * second_momentum_conv_biases[i][j] + (1.0 - beta1) * pow(filterBiases[i][j] / batchSize, 2);

				conv_layers[i].biases[j] = conv_layers[i].biases[j]	- alpha * ((first_momentum_conv_biases[i][j] / corr1) / (sqrt(second_momentum_conv_biases[i][j] / corr2) + EPSILON));
			}
		}
	}

	/**
	 * Description
	 *
	 * @param weightGradient
	 * @param weightBiases
	 * @return
	*/
	void updateWeights(vector<vector<float>> &weightGradient, vector<float> &weightBiases) {
		float corr1 = 1.0f;// - pow(beta1, step); //Korrekturterm des erstes Moments
		float corr2 = 1.0f;// - pow(beta2, step); //Korrekturterm des zweiten Moments

		for (unsigned i = 0; i < first_momentum_weights.size(); i++) {
			for (unsigned j = 0; j < first_momentum_weights[i].size(); j++) {
				first_momentum_weights[i][j] = beta1 * first_momentum_weights[i][j] + (1.0 - beta1) * weightGradient[i][j] / batchSize;
				second_momentum_weights[i][j] = beta1 * second_momentum_weights[i][j] + (1.0 - beta1) * pow(weightGradient[i][j] / batchSize, 2);

				(*connected_layer).weights[i][j] = (*connected_layer).weights[i][j]	- alpha * ((first_momentum_weights[i][j] / corr1) / (sqrt(second_momentum_weights[i][j] / corr2) + EPSILON));
			}
			first_momentum_conn_biases[i] = beta1 * first_momentum_conn_biases[i] + (1.0 - beta1) * weightBiases[i] / batchSize;
			second_momentum_conn_biases[i] = beta1 * second_momentum_conn_biases[i] + (1.0 - beta1) * pow(weightBiases[i] / batchSize, 2);

			(*connected_layer).biases[i] = (*connected_layer).biases[i]	- alpha * ((first_momentum_conn_biases[i] / corr1) / (sqrt(second_momentum_conn_biases[i] / corr2) + EPSILON));
		}
	}
};


////////
// Unnötige Methoden von SGD
///////
/*void updateWeights2(float alpha, vector<vector<float>> &weightGradient, vector<float> &weightBiases, int batchSize) {
		for (unsigned i = 0; i < weightGradient.size(); i++) {
			for (unsigned j = 0; j < weightGradient[i].size(); j++) {
				(*connected_layer).weights[i][j] -= alpha * weightGradient[i][j] / batchSize;
			}
			(*connected_layer).biases[i] -= alpha * weightBiases[i] / batchSize;
		}
	}

	void updateFilters2(float alpha, vector<vector<vector<vector<float>>>> &filterGradients, vector<vector<float>> &filterBiases, int batchSize) {
		for (unsigned i = 0; i < first_momentum_filters.size(); i++) {
			for (unsigned j = 0; j < first_momentum_filters[i].size(); j++) {
				for (unsigned k = 0; k < first_momentum_filters[i][j].size(); k++) {
					for (unsigned l = 0; l < first_momentum_filters[i][j][k].size(); l++) {
						conv_layers[i].filters[j][k][l] -= alpha * filterGradients[i][j][k][l] / batchSize;
					}
				}
				conv_layers[i].biases[j] -= alpha * filterBiases[i][j] / batchSize;
			}
		}
	}*/
