/*
 * FullyConnectedLayer.h
 *
 *  Created on: 24.06.2021
 *      Author: Stefan
 */

#ifndef FULLYCONNECTEDLAYER_H_
#define FULLYCONNECTEDLAYER_H_

#include <mutex>
#include <queue>
#include <random>

#include "ParallelStuff.h"
#include "parameter.h"

class FullyConnectedLayer {
	static const int num_inputs = num_filters;
	static const int input_size1 = imageSizeX_afterPooling;
	static const int input_size2 = imageSizeY_afterPooling;
	/*static const int packetSize		= (num_weights * num_lastLayer_inputNeurons) / num_packets; //groesse der arbeitspakete
	 static const bool needCleanup	= (num_weights * num_lastLayer_inputNeurons) % num_packets != 0; //soll JobCleanup aufgerufen werden?*/

public:

	float (*weights)[num_lastLayer_inputNeurons]; //index1->Klassifikationklasse, index2->gewicht der Klassifikationklasse index1
	float (*biases); //index1->Klassifikationklasse

	//index0->thread
	float (*output)[num_weights]; //index1->Klassifikationklasse
	float (*weight_gradient)[num_weights][num_lastLayer_inputNeurons]; //index1->Klassifikationklasse, index2->gewicht der Klassifikationklasse index1
	float (*bias_gradient)[num_weights]; //index1->Klassifikationklasse
	float (*loss_input)[num_inputs][input_size1][input_size2]; //index1->featureMap (num_inputs viele), index2&3-> x und y der FeatureMap
	//std::deque<std::mutex> mtx; //benoetigt fuer einige parallele aufteilungen, da ueberschneidungen von indizes der arbeitspakete passieren koennen
	float (*loss_gradient)[num_weights]; //pointer auf den loss gradienten (backprop param) (kann theoretisch auch nur einmal gesetzt werden, da pointer danach immer gleich bleibt)
	//index1->featureMap, index2&3-> x und y der FeatureMap
	float (*input)[num_filters][input_size1][input_size2]; //pointer auf den input (forward param) (kann theoretisch auch nur einmal gesetzt werden, da pointer danach immer gleich bleibt)
	//index1->Klassifikationklasse, index2->gewicht der Klassifikationklasse index1

	FullyConnectedLayer();

	void setLossGradient(float loss_gradientP[threads][num_weights]);

	void setInput(float inputP[threads][num_filters][input_size1][input_size2]);

	/*void forwardJob(int packet);

	 void forwardJobCleanup(int packet);*/

	void forward(int_fast8_t spot);

	/*void forward_par(float inputP [num_filters] [input_size1] [input_size2]);

	 void backpropJob(int packet);

	 void backpropJobCleanup(int packet);*/

	void backprop(int_fast8_t spot);

	//void backprop_par(float loss_gradientP [num_weights]);

	void cleanup();
};

#endif /* FULLYCONNECTEDLAYER_H_ */
