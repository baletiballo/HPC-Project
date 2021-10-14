/*
 * cnn.h
 *
 *  Created on: 24.06.2021
 *      Author: Stefan
 */

#ifndef CNN_H_
#define CNN_H_

#include <vector>
#include <random>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <tgmath.h>

#include <mutex>

#include "parameter.h"
#include "FullyConnectedLayer.h"
#include "Conv.h"
#include "MaxPool.h"
#include "ParallelStuff.h"

class CNN {

	
public:

	/*const int packetSizeConv = (num_conv_layers * num_filters) / num_packets; //groesse der arbeitspakete fuer update Conv
	int packetSizeFull = (num_classes * num_finalImages * imageSizeX_afterPooling * imageSizeY_afterPooling) / num_packets; //groesse der arbeitspakete fuer update FullyConnectedLayer
	bool needCleanup = (num_conv_layers * num_filters) % num_packets != 0 || (num_classes * num_finalImages * imageSizeX_afterPooling *imageSizeY_afterPooling) % num_packets != 0; //soll JobCleanup aufgerufen werden?*/

	float corr1 = 1.0f;		//Korrekturterm des erstes Moments
	float corr2 = 1.0f;		//Korrekturterm des zweiten Moments
	int num_updateJobs = num_filters + num_classes; //Anzahl an Jobpaketen, in die update() aufgeteilt ist

	int step; //Anzahl der bisher gelernten Batches
	

	Conv *convLayer; //Vektor aller Convolutional Layers
	MaxPool *poolLayer; //Vektor aller Pooling Layers
	FullyConnectedLayer *connected_layer; //Das eine Fully Connected layer
	
	//Daten für ADAM
	float (*first_momentum_filters)  [conv_size1] [conv_size2]; //Erstes Moment der Filter: index1->Layer, index2->Filter des Layers index1, index3&4-> x bzw y Richtung des Filters
	float (*second_momentum_filters) [conv_size1] [conv_size2]; //Zweites Moment der Filter: index1->Layer, index2->Filter des Layers index1, index3&4-> x bzw y Richtung des Filters
	float (*first_momentum_conv_biases); //Erstes Moment der Filterbiasse: index1->Layer, index2->Filter des Layers index1
	float (*second_momentum_conv_biases); //Zweites Moment der Filterbiasse: index1->Layer, index2->Filter des Layers index1
	float (*first_momentum_weights)  [num_lastLayer_inputNeurons];  //Erstes Moment der Gewichte: index1->Klassifikationklasse, index2->Gewichte der Pixel für Klassifikationklasse index1
	float (*second_momentum_weights) [num_lastLayer_inputNeurons]; //Zweites Moment der Gewichte: index1->Klassifikationklasse, index2->Gewichte der Pixel für Klassifikationklasse index1
	float (*first_momentum_conn_biases); //Erstes Moment der Gewichtbiasse: index1->Klassifikationklasse
	float (*second_momentum_conn_biases); //Zweites Moment der Gewichtbiasse: index1->Klassifikationklasse

	std::mutex mtx;
	int totalCorrect;
	float totalLoss;

	int_fast8_t (*labels);

	CNN();

	void forward(int image, int_fast8_t spot);

	std::tuple<float, int> learn(float (*x_batch) [imageSizeX] [imageSizeY], int_fast8_t (*y_batch));

	/*void updateJob(int packet);

	void updateJobCleanup(int packet);

	void update_par();*/

	void update();
	void update_par(int job);
	void softmax(float t1 [num_classes]);
	void ReLu(float t1 [num_filters] [imageSizeX_afterConvolution] [imageSizeY_afterConvolution]);
	void ReLuPrime( float t1 [num_filters] [imageSizeX_afterConvolution] [imageSizeY_afterConvolution],
                float t2  [num_filters] [imageSizeX_afterConvolution] [imageSizeY_afterConvolution]);

};

#endif /* CNN_H_ */
