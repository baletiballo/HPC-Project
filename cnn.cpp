#include "cnn.h"

CNN::CNN() {
	step = 1;
	totalLoss = 0;
	totalCorrect = 0;

	convLayer = new Conv();
	poolLayer = new MaxPool();
	connected_layer = new FullyConnectedLayer();
	convLayer->setLossGradient(poolLayer->loss_input);
	poolLayer->setInput(convLayer->output);
	poolLayer->setLossGradient(connected_layer->loss_input);
	connected_layer->setInput(poolLayer->output);
	connected_layer->setLossGradient(connected_layer->output);

	first_momentum_filters = new float[num_filters][conv_size1][conv_size2] { };
	second_momentum_filters = new float[num_filters][conv_size1][conv_size2] { };
	first_momentum_conv_biases = new float[num_filters] { };
	second_momentum_conv_biases = new float[num_filters] { };
	first_momentum_weights = new float[num_classes][num_lastLayer_inputNeurons] { };
	second_momentum_weights = new float[num_classes][num_lastLayer_inputNeurons] { };
	first_momentum_conn_biases = new float[num_classes] { };
	second_momentum_conn_biases = new float[num_classes] { };
}

/**
 * Jagt image durch das Netzwerk und vergleicht das Ergebnis mit label
 *
 * @param image Das Bild, dass klassifiziert werden soll
 * @param label Das korrekte Lable
 * @return Das Tupel (loss, correct)
 */
void CNN::forward(int image, int_fast8_t spot) {
	int_fast8_t label = labels[image];

	convLayer->forward(spot, image); //bild in ersten conv layer
	ReLu(convLayer->output[spot]);
	poolLayer->forward(spot);
	connected_layer->forward(spot); //nutzt immer den output des letzten pooling layers als input

	softmax(connected_layer->output[spot]); //achtung alles folgende findet in place statt, es aendert sich also connected_layer->output
	float loss = -log(connected_layer->output[spot][label]);

	int argmax = 0;
	for (int i = 0; i < num_classes; i++)
		if (connected_layer->output[spot][i] >= connected_layer->output[spot][argmax])
			argmax = i;

	connected_layer->output[spot][label] -= 1;

	connected_layer->backprop(spot); //transformierter connected_layer->output wird als input fuer backprop genutzt

	poolLayer->backprop(spot); //zurueckgehen in umgekehrter Reihenfolge
	ReLuPrime(poolLayer->loss_input[spot], poolLayer->input[spot]);
	convLayer->backprop(spot, image);

	mtx.lock();
	totalLoss += loss;
	if (label == argmax) {
		totalCorrect++;
	}
	mtx.unlock();
	sem.V(1);
}

/**
 * Lernzyklus fuer ein batch
 *
 * @param images Pointer auf den Beginn des Batches in den Trainingsdaten
 * @param lables Pointer auf den Beginn des Batches in den Trainingslabels
 * @return
 */
std::tuple<float, int> CNN::learn(float (*x_batch) [imageSizeX][imageSizeY], int_fast8_t (*y_batch)) {
	convLayer->setInput(x_batch);
	labels = y_batch;

	totalLoss = 0.0f;
	totalCorrect = 0;

	sem.set(0);
	for (int i = 0; i < batchSize; i++) {
		pushJob(i);
	}
	sem.P(batchSize);
	//ADAM learning
	update();
	step++;

	return {totalLoss, totalCorrect};
}


//void CNN::updateJob(int packet) {
//	for (int index = packet * packetSizeConv; index < (packet + 1) * packetSizeConv; index++) {
//		int currFilter = index % num_filters;
//
//		for (int currX = 0; currX < conv_size1; currX++) {
//			for (int currY = 0; currY < conv_size2; currY++) {
//				first_momentum_filters[currFilter][currX][currY] = beta1 * first_momentum_filters[currFilter][currX][currY]
//					+ (1.0f - beta1) * (convLayer->filter_gradient)[currFilter][currX][currY] / batchSize; //erstes moment der filter updaten
//				second_momentum_filters[currFilter][currX][currY] = beta1 * second_momentum_filters[currFilter][currX][currY]
//					+ (1.0f - beta1) * powf((convLayer->filter_gradient)[currFilter][currX][currY] / batchSize, 2); //zweites moment der filter updaten
//
//				(convLayer->filter_gradient)[currFilter][currX][currY] = 0.0; //gradienten wieder auf 0 zuruecksetzen fuer naechstes batch
//
//				convLayer->filters[currFilter][currX][currY] = convLayer->filters[currFilter][currX][currY]
//					- alpha	* ((first_momentum_filters[currFilter][currX][currY] / corr1)
//					/ (sqrt(second_momentum_filters[currFilter][currX][currY] / corr2) + EPSILON)); //filter des conv layers updaten
//			}
//		}
//
//		first_momentum_conv_biases[currFilter] = beta1 * first_momentum_conv_biases[currFilter]
//			+ (1.0f - beta1) * (convLayer->bias_gradient)[currFilter] / batchSize; //erstes moment der filterbiasse updaten
//		second_momentum_conv_biases[currFilter] = beta1 * second_momentum_conv_biases[currFilter]
//			+ (1.0f - beta1) * powf((convLayer->bias_gradient)[currFilter] / batchSize, 2); //zweites moment der filterbiasse updaten
//
//		(convLayer->bias_gradient)[currFilter] = 0.0; //Gradienten wieder auf 0 zuruecksetzen fuer naechstes batch
//
//		convLayer->biases[currFilter] = convLayer->biases[currFilter]
//			- alpha * ((first_momentum_conv_biases[currFilter] / corr1)
//			/ (sqrt(second_momentum_conv_biases[currFilter] / corr2) + EPSILON)); //filterbiasse des conv layers updaten
//	}
//
//	for (int index = packet * packetSizeFull; index < (packet + 1) * packetSizeFull; index++) {
//		int currClass = index / num_lastLayer_inputNeurons;
//		int currWeight = index % num_lastLayer_inputNeurons;
//
//		first_momentum_weights[currClass][currWeight] = beta1 * first_momentum_weights[currClass][currWeight]
//			+ (1.0f - beta1) * (connected_layer->weight_gradient)[currClass][currWeight] / batchSize; //erstes moment der fcl gewichte updaten
//		second_momentum_weights[currClass][currWeight] = beta1 * second_momentum_weights[currClass][currWeight]
//			+ (1.0f - beta1) * powf((connected_layer->weight_gradient)[currClass][currWeight] / batchSize, 2); //zweites moment der fcl gewichte updaten
//
//		(connected_layer->weight_gradient)[currClass][currWeight] = 0.0f; //Gradienten wieder auf 0 zuruecksetzen fuer naechstes batch
//
//		connected_layer->weights[currClass][currWeight] = connected_layer->weights[currClass][currWeight]
//			- alpha * ((first_momentum_weights[currClass][currWeight] / corr1) / (sqrt(second_momentum_weights[currClass][currWeight] / corr2) + EPSILON)); //gewichte des fcl updaten
//	}
//	sem.V(1);
//}
//
//void CNN::updateJobCleanup(int packet) {
//	for (int index = packet * packetSizeConv; index < (num_conv_layers * num_filters); index++) {
//		int currFilter = index % num_filters;
//
//		for (int currX = 0; currX < conv_size1; currX++) {
//			for (int currY = 0; currY < conv_size2; currY++) {
//				first_momentum_filters[currFilter][currX][currY] = beta1 * first_momentum_filters[currFilter][currX][currY]
//					+ (1.0f - beta1) * (convLayer->filter_gradient)[currFilter][currX][currY] / batchSize; //erstes moment der filter updaten
//				second_momentum_filters[currFilter][currX][currY] = beta1 * second_momentum_filters[currFilter][currX][currY]
//					+ (1.0f - beta1) * powf((convLayer->filter_gradient)[currFilter][currX][currY] / batchSize, 2); //zweites moment der filter updaten
//
//				(convLayer->filter_gradient)[currFilter][currX][currY] = 0.0; //gradienten wieder auf 0 zuruecksetzen fuer naechstes batch
//
//				convLayer->filters[currFilter][currX][currY] = convLayer->filters[currFilter][currX][currY]
//					- alpha * ((first_momentum_filters[currFilter][currX][currY] / corr1)
//					/ (sqrt(second_momentum_filters[currFilter][currX][currY] / corr2) + EPSILON)); //filter des conv layers updaten
//			}
//		}
//
//		first_momentum_conv_biases[currFilter] = beta1 * first_momentum_conv_biases[currFilter]
//			+ (1.0f - beta1) * (convLayer->bias_gradient)[currFilter] / batchSize; //erstes moment der filterbiasse updaten
//		second_momentum_conv_biases[currFilter] = beta1 * second_momentum_conv_biases[currFilter]
//			+ (1.0f - beta1) * powf((convLayer->bias_gradient)[currFilter] / batchSize, 2); //zweites moment der filterbiasse updaten
//
//		(convLayer->bias_gradient)[currFilter] = 0.0; //gradienten wieder auf 0 zuruecksetzen fuer naechstes batch
//
//		convLayer->biases[currFilter] = convLayer->biases[currFilter]
//			- alpha * ((first_momentum_conv_biases[currFilter] / corr1)
//			/ (sqrt(second_momentum_conv_biases[currFilter] / corr2) + EPSILON)); //filterbiasse des conv layers updaten
//	}
//
//	for (unsigned index = packet * packetSizeFull; index < num_classes * num_lastLayer_inputNeurons; index++) {
//		int currClass = index / num_lastLayer_inputNeurons;
//		int currWeight = index % num_lastLayer_inputNeurons;
//
//		first_momentum_weights[currClass][currWeight] = beta1 * first_momentum_weights[currClass][currWeight]
//			+ (1.0f - beta1) * (connected_layer->weight_gradient)[currClass][currWeight] / batchSize; //erstes moment der fcl gewichte updaten
//		second_momentum_weights[currClass][currWeight] = beta1 * second_momentum_weights[currClass][currWeight]
//			+ (1.0f - beta1) * powf((connected_layer->weight_gradient)[currClass][currWeight] / batchSize, 2); //zweites moment der fcl gewichte updaten
//
//		(connected_layer->weight_gradient)[currClass][currWeight] = 0.0; //gradienten wieder auf 0 zuruecksetzen fuer naechstes batch
//
//		connected_layer->weights[currClass][currWeight] = connected_layer->weights[currClass][currWeight]
//			- alpha * ((first_momentum_weights[currClass][currWeight] / corr1)
//			/ (sqrt(second_momentum_weights[currClass][currWeight] / corr2) + EPSILON)); //gewichte des fcl updaten
//	}
//}
//
//void CNN::update_par() {
//	corr1 = 1.0f;		// - pow(beta1, step); //Korrekturterm des erstes Moments
//	corr2 = 1.0f;		// - pow(beta2, step); //Korrekturterm des zweiten Moments
//
//	sem.set(0);
//	//pool.setCNN(*this);
//	pool.setTask(9);
//	for (int i = 0; i < num_packets; i++)
//		pushJob(i);
//
//	if (needCleanup)
//		updateJobCleanup(num_packets + 1);
//
//	for (int currClass = 0; currClass < num_classes; currClass++) {
//		first_momentum_conn_biases[currClass] = beta1 * first_momentum_conn_biases[currClass]
//			+ (1.0f - beta1) * (connected_layer->bias_gradient)[currClass] / batchSize; //erstes moment der fcl biasse updaten
//		second_momentum_conn_biases[currClass] = beta1 * second_momentum_conn_biases[currClass]
//			+ (1.0f - beta1) * powf((connected_layer->bias_gradient)[currClass] / batchSize, 2); //zweites moment der fcl biasse updaten
//
//		(connected_layer->bias_gradient)[currClass] = 0.0; //gradienten wieder auf 0 zuruecksetzen fuer naechstes batch
//
//		connected_layer->biases[currClass] = connected_layer->biases[currClass]
//			- alpha * ((first_momentum_conn_biases[currClass] / corr1)
//			/ (sqrt(second_momentum_conn_biases[currClass] / corr2) + EPSILON)); //biasse des fcl updaten
//	}
//
//	sem.P(num_packets);
//}

/**
 * Update der Momente und Gewichte und Filter
 *
 */
void CNN::update() {
	corr1 = 1.0f;		// - pow(beta1, step); //Korrekturterm des erstes Moments
	corr2 = 1.0f;		// - pow(beta2, step); //Korrekturterm des zweiten Moments

	for (unsigned curr_filter = 0; curr_filter < num_filters; curr_filter++) { //filters des conv layer
		for (unsigned k = 0; k < conv_size1; k++) { //filter x
			for (unsigned l = 0; l < conv_size2; l++) { //filter y

				float fg = 0.0;
				for (int spot = 0; spot < threads; spot++) {
					fg += (convLayer->filter_gradient)[spot][curr_filter][k][l];
				}

				first_momentum_filters[curr_filter][k][l] = beta1 * first_momentum_filters[curr_filter][k][l] + (1.0f - beta1) * fg / batchSize; //erstes moment der filter updaten
				second_momentum_filters[curr_filter][k][l] = beta2 * second_momentum_filters[curr_filter][k][l] + (1.0f - beta2) * powf(fg / batchSize, 2); //zweites moment der filter updaten

				convLayer->filters[curr_filter][k][l] = convLayer->filters[curr_filter][k][l]
						- alpha * ((first_momentum_filters[curr_filter][k][l] / corr1) / (sqrt(second_momentum_filters[curr_filter][k][l] / corr2) + EPSILON)); //filter des conv layers updaten
			}
		}

		float bg = 0.0;
		for (int spot = 0; spot < threads; spot++) {
			bg += (convLayer->bias_gradient)[spot][curr_filter];
		}

		first_momentum_conv_biases[curr_filter] = beta1 * first_momentum_conv_biases[curr_filter] + (1.0f - beta1) * bg / batchSize; //erstes moment der filterbiasse updaten
		second_momentum_conv_biases[curr_filter] = beta2 * second_momentum_conv_biases[curr_filter] + (1.0f - beta2) * powf(bg / batchSize, 2); //zweites moment der filterbiasse updaten

		convLayer->biases[curr_filter] = convLayer->biases[curr_filter]
				- alpha * ((first_momentum_conv_biases[curr_filter] / corr1) / (sqrt(second_momentum_conv_biases[curr_filter] / corr2) + EPSILON)); //filterbiasse des conv layers updaten
	}

	convLayer->cleanup(); //gradienten wieder auf 0 zuruecksetzen fuer naechstes batch
	for (unsigned curr_class = 0; curr_class < num_classes; curr_class++) { //outputzahlen
		for (unsigned curr_neuron = 0; curr_neuron < num_lastLayer_inputNeurons; curr_neuron ++) { //gesamtgroesse des inputs f�r den fully connected layer (fcl)

			float wg = 0.0;
			for (int spot = 0; spot < threads; spot++) {
				wg += (connected_layer->weight_gradient)[spot][curr_class][curr_neuron];
			}

			first_momentum_weights[curr_class][curr_neuron + 0] = beta1 * first_momentum_weights[curr_class][curr_neuron + 0] + (1.0f - beta1) * wg / batchSize; //erstes moment der fcl gewichte updaten
			second_momentum_weights[curr_class][curr_neuron + 0] = beta2 * second_momentum_weights[curr_class][curr_neuron + 0]
					+ (1.0f - beta2) * powf(wg / batchSize, 2); //zweites moment der fcl gewichte updaten
			connected_layer->weights[curr_class][curr_neuron + 0] = connected_layer->weights[curr_class][curr_neuron + 0]
					- alpha
							* ((first_momentum_weights[curr_class][curr_neuron + 0] / corr1)
									/ (sqrt(second_momentum_weights[curr_class][curr_neuron + 0] / corr2) + EPSILON)); //gewichte des fcl updaten
		}

		float bg = 0.0;
		for (int spot = 0; spot < threads; spot++) {
			bg += (connected_layer->bias_gradient)[spot][curr_class];
		}

		first_momentum_conn_biases[curr_class] = beta1 * first_momentum_conn_biases[curr_class] + (1.0f - beta1) * bg / batchSize; //erstes moment der fcl biasse updaten
		second_momentum_conn_biases[curr_class] = beta2 * second_momentum_conn_biases[curr_class] + (1.0f - beta2) * powf(bg / batchSize, 2); //zweites moment der fcl biasse updaten

		connected_layer->biases[curr_class] = connected_layer->biases[curr_class]
				- alpha * ((first_momentum_conn_biases[curr_class] / corr1) / (sqrt(second_momentum_conn_biases[curr_class] / corr2) + EPSILON)); //biasse des fcl updaten
	}

	connected_layer->cleanup(); //gradienten wieder auf 0 zuruecksetzen fuer naechstes batch
}



////////
// Unnötige Methoden von SGD
///////
/*void updateWeights2(float alpha, vector<vector<float>> &weightGradient, vector<float> &weightBiases, int batchSize) {
 for (unsigned i = 0; i < weightGradient.size(); i++) {
 for (unsigned j = 0; j < weightGradient[i].size(); j++) {
 connected_layer->weights[i][j] -= alpha * weightGradient[i][j] / batchSize;
 }
 connected_layer->biases[i] -= alpha * weightBiases[i] / batchSize;
 }
 }

 void updateFilters2(float alpha, vector<vector<vector<vector<float>>>> &filterGradients, vector<vector<float>> &filterBiases, int batchSize) {
 for (unsigned i = 0; i < first_momentum_filters.size(); i++) {
 for (unsigned j = 0; j < first_momentum_filters.size(); j++) {
 for (unsigned k = 0; k < first_momentum_filters[j].size(); k++) {
 for (unsigned l = 0; l < first_momentum_filters[j][k].size(); l++) {
 convLayers[i].filters[j][k][l] -= alpha * filterGradients[i][j][k][l] / batchSize;
 }
 }
 convLayers[i].biases[j] -= alpha * filterBiases[i][j] / batchSize;
 }
 }
 }*/
