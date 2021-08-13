#include "cnn.h"
#include <random>
#include "ParallelStuff.h"
#include "Hilfsfunktionen.h"
#include "ReLu.h"
#include "parameter.h"

#include <fstream>
#include <iostream>
#include <sstream>

CNN::CNN(int imageSize) :
		imageSize(imageSize) {
	step = 1;

	int currX = imageSize;
	int currY = imageSize;
	int images = 1;
	for (int i = 0; i < num_conv_layers; i++) {
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

	packetSizeConv = (num_conv_layers * num_filters) / packets;
	packetSizeFull = (num_weights * images * currX * currY) / packets;
	needCleanup = (num_conv_layers * num_filters) % packets != 0 || (num_weights * images * currX * currY) % packets != 0;
}

/**
 * Jagt @Param image durch das Netzwerk und vergleicht das Ergebnis mit @Param lable
 *
 * @param image
 * @param label
 * @return Das Tupel (loss, correct)
 */
tuple<float, bool> CNN::forward(vector<vector<vector<float>>> &image, int_fast8_t label) {

	conv_layers[0].forward(image); //bild in ersten conv layer
	ReLu(conv_layers[0].output);
	pooling_layers[0].forward(conv_layers[0].output);

	for (int i = 1; i < num_conv_layers; i++) { //alle weiteren conv+pooling layers
		conv_layers[i].forward(pooling_layers[i - 1].output); //nutzt immer den output des letzten pooling layers als input
		ReLu(conv_layers[i].output);
		pooling_layers[i].forward(conv_layers[i].output);
	}

	(*connected_layer).forward(pooling_layers[num_conv_layers - 1].output); //nutzt immer den output des letzten pooling layers als input

	softmax((*connected_layer).output); //achtung alles folgende findet in place statt, es aendert sich als (*connected_layer).output
	float loss = -log((*connected_layer).output[label]);

	int argmax = 0;
	for (int i = 0; i < num_weights; i++)
		if ((*connected_layer).output[i] >= (*connected_layer).output[argmax])
			argmax = i;

	bool correct = label == argmax;
	(*connected_layer).output[label] -= 1;

	(*connected_layer).backprop((*connected_layer).output); //transformierter (*connected_layer).output wird als input fuer backprop genutzt

	pooling_layers[num_conv_layers - 1].backprop((*connected_layer).loss_input); //zurueckgehen in umgekehrter Reihenfolge
	ReLuPrime(pooling_layers[num_conv_layers - 1].loss_input, *pooling_layers[num_conv_layers - 1].input);
	conv_layers[num_conv_layers - 1].backprop(pooling_layers[num_conv_layers - 1].loss_input);

	for (int i = num_conv_layers - 2; i > -1; i--) { //zurueckgehen in umgekehrter Reihenfolge
		pooling_layers[i].backprop(conv_layers[i + 1].loss_input); //nutzt immer den output des darauf folgenden conv layers als input
		ReLuPrime(pooling_layers[i].loss_input, *pooling_layers[i].input);
		conv_layers[i].backprop(pooling_layers[i].loss_input);
	}

	return {loss, correct};
}

/**
 * Lernzyklus fuer ein batch
 *
 * @param x_batch
 * @param y_batch
 * @return
 */
tuple<float, int_fast8_t> CNN::learn(vector<vector<vector<float>> > &x_batch, vector<int_fast8_t> &y_batch) {

	//Das Aktuell zu verarbeitende Bild. Als einelementiger Vektor, da Conv-Layer Vektoren von Bildern nehmen
	vector<vector<vector<float>>> image(1, vector<vector<float>>(imageSize, vector<float>(imageSize)));
	image[0] = x_batch[0];
	int label = y_batch[0]; //Das Lable des aktuellen Bildes
	//Ergebnis des Netzwerks nach dem Forward von image
	tuple<float, bool> Result = forward(image, label);

	float loss = get<0>(Result);
	int_fast8_t correct = get<1>(Result);

	for (int i = 1; i < batchSize; i++) {
		image[0] = x_batch[i];
		label = y_batch[i];

		Result = forward(image, label);
		loss += get<0>(Result);
		correct += get<1>(Result);
	}

	//ADAM learning
	update();
	step++;

	return {loss, correct};
}

void CNN::updateJob(int packet) {
	for (int index = packet * packetSizeConv; index < (packet + 1) * packetSizeConv; index++) {
		int currConvLayer = index / num_filters;
		int currFilter = index % num_filters;

		for (int currX = 0; currX < conv_size1; currX++) {
			for (int currY = 0; currY < conv_size2; currY++) {
				first_momentum_filters[currConvLayer][currFilter][currX][currY] = beta1 * first_momentum_filters[currConvLayer][currFilter][currX][currY]
						+ (1.0 - beta1) * (conv_layers[currConvLayer].filter_gradient)[currFilter][currX][currY] / batchSize; //erstes moment der filter updaten
				second_momentum_filters[currConvLayer][currFilter][currX][currY] = beta1 * second_momentum_filters[currConvLayer][currFilter][currX][currY]
						+ (1.0 - beta1) * pow((conv_layers[currConvLayer].filter_gradient)[currFilter][currX][currY] / batchSize, 2); //zweites moment der filter updaten

				(conv_layers[currConvLayer].filter_gradient)[currFilter][currX][currY] = 0.0; //gradienten wieder auf 0 zuruecksetzen fuer naechstes batch

				conv_layers[currConvLayer].filters[currFilter][currX][currY] = conv_layers[currConvLayer].filters[currFilter][currX][currY]
						- alpha
								* ((first_momentum_filters[currConvLayer][currFilter][currX][currY] / corr1)
										/ (sqrt(second_momentum_filters[currConvLayer][currFilter][currX][currY] / corr2) + EPSILON)); //filter des conv layers updaten
			}
		}

		first_momentum_conv_biases[currConvLayer][currFilter] = beta1 * first_momentum_conv_biases[currConvLayer][currFilter]
				+ (1.0 - beta1) * (conv_layers[currConvLayer].bias_gradient)[currFilter] / batchSize; //erstes moment der filterbiasse updaten
		second_momentum_conv_biases[currConvLayer][currFilter] = beta1 * second_momentum_conv_biases[currConvLayer][currFilter]
				+ (1.0 - beta1) * pow((conv_layers[currConvLayer].bias_gradient)[currFilter] / batchSize, 2); //zweites moment der filterbiasse updaten

		(conv_layers[currConvLayer].bias_gradient)[currFilter] = 0.0; //gradienten wieder auf 0 zuruecksetzen fuer naechstes batch

		conv_layers[currConvLayer].biases[currFilter] = conv_layers[currConvLayer].biases[currFilter]
				- alpha
						* ((first_momentum_conv_biases[currConvLayer][currFilter] / corr1)
								/ (sqrt(second_momentum_conv_biases[currConvLayer][currFilter] / corr2) + EPSILON)); //filterbiasse des conv layers updaten
	}

	for (int index = packet * packetSizeFull; index < (packet + 1) * packetSizeFull; index++) {
		int currClass = index / first_momentum_weights[0].size();
		int currWeight = index % first_momentum_weights[0].size();

		first_momentum_weights[currClass][currWeight] = beta1 * first_momentum_weights[currClass][currWeight]
				+ (1.0 - beta1) * ((*connected_layer).weight_gradient)[currClass][currWeight] / batchSize; //erstes moment der fcl gewichte updaten
		second_momentum_weights[currClass][currWeight] = beta1 * second_momentum_weights[currClass][currWeight]
				+ (1.0 - beta1) * pow(((*connected_layer).weight_gradient)[currClass][currWeight] / batchSize, 2); //zweites moment der fcl gewichte updaten

		((*connected_layer).weight_gradient)[currClass][currWeight] = 0.0; //gradienten wieder auf 0 zuruecksetzen fuer naechstes batch

		(*connected_layer).weights[currClass][currWeight] = (*connected_layer).weights[currClass][currWeight]
				- alpha * ((first_momentum_weights[currClass][currWeight] / corr1) / (sqrt(second_momentum_weights[currClass][currWeight] / corr2) + EPSILON)); //gewichte des fcl updaten
	}
	sem.V(1);
}

void CNN::updateJobCleanup(int packet) {
	for (int index = packet * packetSizeConv; index < (num_conv_layers * num_filters); index++) {
		int currConvLayer = index / num_filters;
		int currFilter = index % num_filters;

		for (int currX = 0; currX < conv_size1; currX++) {
			for (int currY = 0; currY < conv_size2; currY++) {
				first_momentum_filters[currConvLayer][currFilter][currX][currY] = beta1 * first_momentum_filters[currConvLayer][currFilter][currX][currY]
						+ (1.0 - beta1) * (conv_layers[currConvLayer].filter_gradient)[currFilter][currX][currY] / batchSize; //erstes moment der filter updaten
				second_momentum_filters[currConvLayer][currFilter][currX][currY] = beta1 * second_momentum_filters[currConvLayer][currFilter][currX][currY]
						+ (1.0 - beta1) * pow((conv_layers[currConvLayer].filter_gradient)[currFilter][currX][currY] / batchSize, 2); //zweites moment der filter updaten

				(conv_layers[currConvLayer].filter_gradient)[currFilter][currX][currY] = 0.0; //gradienten wieder auf 0 zuruecksetzen fuer naechstes batch

				conv_layers[currConvLayer].filters[currFilter][currX][currY] = conv_layers[currConvLayer].filters[currFilter][currX][currY]
						- alpha
								* ((first_momentum_filters[currConvLayer][currFilter][currX][currY] / corr1)
										/ (sqrt(second_momentum_filters[currConvLayer][currFilter][currX][currY] / corr2) + EPSILON)); //filter des conv layers updaten
			}
		}

		first_momentum_conv_biases[currConvLayer][currFilter] = beta1 * first_momentum_conv_biases[currConvLayer][currFilter]
				+ (1.0 - beta1) * (conv_layers[currConvLayer].bias_gradient)[currFilter] / batchSize; //erstes moment der filterbiasse updaten
		second_momentum_conv_biases[currConvLayer][currFilter] = beta1 * second_momentum_conv_biases[currConvLayer][currFilter]
				+ (1.0 - beta1) * pow((conv_layers[currConvLayer].bias_gradient)[currFilter] / batchSize, 2); //zweites moment der filterbiasse updaten

		(conv_layers[currConvLayer].bias_gradient)[currFilter] = 0.0; //gradienten wieder auf 0 zuruecksetzen fuer naechstes batch

		conv_layers[currConvLayer].biases[currFilter] = conv_layers[currConvLayer].biases[currFilter]
				- alpha
						* ((first_momentum_conv_biases[currConvLayer][currFilter] / corr1)
								/ (sqrt(second_momentum_conv_biases[currConvLayer][currFilter] / corr2) + EPSILON)); //filterbiasse des conv layers updaten
	}

	for (unsigned index = packet * packetSizeFull; index < num_weights * first_momentum_weights[0].size(); index++) {
		int currClass = index / first_momentum_weights[0].size();
		int currWeight = index % first_momentum_weights[0].size();

		first_momentum_weights[currClass][currWeight] = beta1 * first_momentum_weights[currClass][currWeight]
				+ (1.0 - beta1) * ((*connected_layer).weight_gradient)[currClass][currWeight] / batchSize; //erstes moment der fcl gewichte updaten
		second_momentum_weights[currClass][currWeight] = beta1 * second_momentum_weights[currClass][currWeight]
				+ (1.0 - beta1) * pow(((*connected_layer).weight_gradient)[currClass][currWeight] / batchSize, 2); //zweites moment der fcl gewichte updaten

		((*connected_layer).weight_gradient)[currClass][currWeight] = 0.0; //gradienten wieder auf 0 zuruecksetzen fuer naechstes batch

		(*connected_layer).weights[currClass][currWeight] = (*connected_layer).weights[currClass][currWeight]
				- alpha * ((first_momentum_weights[currClass][currWeight] / corr1) / (sqrt(second_momentum_weights[currClass][currWeight] / corr2) + EPSILON)); //gewichte des fcl updaten
	}
}

void CNN::update_par() {
	corr1 = 1.0f;		// - pow(beta1, step); //Korrekturterm des erstes Moments
	corr2 = 1.0f;		// - pow(beta2, step); //Korrekturterm des zweiten Moments

	sem.set(0);
	pool.setCNN(*this);
	pool.setTask(9);
	for (int i = 0; i < packets; i++) {
		pushJob(i);
	}
	if (needCleanup) {
		updateJobCleanup(packets + 1);
	}
	for (int currClass = 0; currClass < num_weights; currClass++) {
		first_momentum_conn_biases[currClass] = beta1 * first_momentum_conn_biases[currClass]
				+ (1.0 - beta1) * ((*connected_layer).bias_gradient)[currClass] / batchSize; //erstes moment der fcl biasse updaten
		second_momentum_conn_biases[currClass] = beta1 * second_momentum_conn_biases[currClass]
				+ (1.0 - beta1) * pow(((*connected_layer).bias_gradient)[currClass] / batchSize, 2); //zweites moment der fcl biasse updaten

		((*connected_layer).bias_gradient)[currClass] = 0.0; //gradienten wieder auf 0 zuruecksetzen fuer naechstes batch

		(*connected_layer).biases[currClass] = (*connected_layer).biases[currClass]
				- alpha * ((first_momentum_conn_biases[currClass] / corr1) / (sqrt(second_momentum_conn_biases[currClass] / corr2) + EPSILON)); //biasse des fcl updaten
	}
	sem.P(packets);
}

/**
 * Update der Momente und Gewichte und Filter
 *
 */
void CNN::update() {
	corr1 = 1.0f;		// - pow(beta1, step); //Korrekturterm des erstes Moments
	corr2 = 1.0f;		// - pow(beta2, step); //Korrekturterm des zweiten Moments

	for (unsigned i = 0; i < first_momentum_filters.size(); i++) { //conv layers
		for (unsigned j = 0; j < first_momentum_filters[i].size(); j++) { //filters des conv layer
			for (unsigned k = 0; k < first_momentum_filters[i][j].size(); k++) { //filter x
				for (unsigned l = 0; l < first_momentum_filters[i][j][k].size(); l++) { //filter y
					first_momentum_filters[i][j][k][l] = beta1 * first_momentum_filters[i][j][k][l]
							+ (1.0 - beta1) * (conv_layers[i].filter_gradient)[j][k][l] / batchSize; //erstes moment der filter updaten
					second_momentum_filters[i][j][k][l] = beta1 * second_momentum_filters[i][j][k][l]
							+ (1.0 - beta1) * pow((conv_layers[i].filter_gradient)[j][k][l] / batchSize, 2); //zweites moment der filter updaten

					conv_layers[i].filters[j][k][l] = conv_layers[i].filters[j][k][l]
							- alpha * ((first_momentum_filters[i][j][k][l] / corr1) / (sqrt(second_momentum_filters[i][j][k][l] / corr2) + EPSILON)); //filter des conv layers updaten
				}
			}
			first_momentum_conv_biases[i][j] = beta1 * first_momentum_conv_biases[i][j] + (1.0 - beta1) * (conv_layers[i].bias_gradient)[j] / batchSize; //erstes moment der filterbiasse updaten
			second_momentum_conv_biases[i][j] = beta1 * second_momentum_conv_biases[i][j]
					+ (1.0 - beta1) * pow((conv_layers[i].bias_gradient)[j] / batchSize, 2); //zweites moment der filterbiasse updaten

			conv_layers[i].biases[j] = conv_layers[i].biases[j]
					- alpha * ((first_momentum_conv_biases[i][j] / corr1) / (sqrt(second_momentum_conv_biases[i][j] / corr2) + EPSILON)); //filterbiasse des conv layers updaten
		}
		conv_layers[i].cleanup(); //gradienten wieder auf 0 zuruecksetzen fuer naechstes batch
	}

	for (unsigned i = 0; i < first_momentum_weights.size(); i++) { //outputzahlen
		for (unsigned j = 0; j < first_momentum_weights[i].size(); j++) { //gesamtgroesse des inputs f�r den fully connected layer (fcl)
			first_momentum_weights[i][j] = beta1 * first_momentum_weights[i][j] + (1.0 - beta1) * ((*connected_layer).weight_gradient)[i][j] / batchSize; //erstes moment der fcl gewichte updaten
			second_momentum_weights[i][j] = beta1 * second_momentum_weights[i][j]
					+ (1.0 - beta1) * pow(((*connected_layer).weight_gradient)[i][j] / batchSize, 2); //zweites moment der fcl gewichte updaten

			(*connected_layer).weights[i][j] = (*connected_layer).weights[i][j]
					- alpha * ((first_momentum_weights[i][j] / corr1) / (sqrt(second_momentum_weights[i][j] / corr2) + EPSILON)); //gewichte des fcl updaten
		}
		first_momentum_conn_biases[i] = beta1 * first_momentum_conn_biases[i] + (1.0 - beta1) * ((*connected_layer).bias_gradient)[i] / batchSize; //erstes moment der fcl biasse updaten
		second_momentum_conn_biases[i] = beta1 * second_momentum_conn_biases[i] + (1.0 - beta1) * pow(((*connected_layer).bias_gradient)[i] / batchSize, 2); //zweites moment der fcl biasse updaten

		(*connected_layer).biases[i] = (*connected_layer).biases[i]
				- alpha * ((first_momentum_conn_biases[i] / corr1) / (sqrt(second_momentum_conn_biases[i] / corr2) + EPSILON)); //biasse des fcl updaten
	}

	(*connected_layer).cleanup(); //gradienten wieder auf 0 zuruecksetzen fuer naechstes batch
}

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
