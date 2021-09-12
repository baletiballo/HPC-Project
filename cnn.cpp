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
	sem.set(0);
	for (int i = 0; i < num_updateJobs; i++) {
		pushJob(i + batchSize); //nicht die schönste Technik einen 2D Input umzusetzen, aber es sollte passen.
	}
	sem.P(num_updateJobs);
	convLayer->cleanup(); //Gradienten wieder auf 0 zuruecksetzen fuer naechstes batch
	connected_layer->cleanup(); //Gradienten wieder auf 0 zuruecksetzen fuer naechstes batch
	
	//update(); //sequenziell updaten

	step++;

	return {totalLoss, totalCorrect};
}


/**
 * Update der Momente und Gewichte und Filter
 * Jeder Job ist entweder ein Filter, oder ein Klassifikationsneuron. Das geht zwar nicht perfekt auf, mit der Anzahl der Threads,
 * aber es erspart synchronisierung und ist nahe dran.
 *
 */
void CNN::update_par(int job) {
	corr1 = 1.0f;		// - pow(beta1, step); //Korrekturterm des erstes Moments
	corr2 = 1.0f;		// - pow(beta2, step); //Korrekturterm des zweiten Moments
	if(job < num_filters) { //Filter des conv layer
		int curr_filter = job;
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

	else { //outputzahlen
		int curr_class = job - num_filters; 
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
	sem.V(1);
	
}




/**
 * Update der Momente und Gewichte und Filter
 *
 */
void CNN::update() {
	corr1 = 1.0f;		// - pow(beta1, step); //Korrekturterm des erstes Moments
	corr2 = 1.0f;		// - pow(beta2, step); //Korrekturterm des zweiten Moments

	for (unsigned curr_filter = 0; curr_filter < num_filters; curr_filter++) { //Filter des conv layer
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