/*
 * Conv.h
 *
 *  Created on: 24.06.2021
 *      Author: Stefan
 */

#ifndef CNN_H_
#define CNN_H_

#include <vector>
#include "FullyConnectedLayer.h"
#include "Conv.h"
#include "MaxPool.h"

using namespace std;

class CNN {
public:
	const int num_conv_layers = 1; //Anzahl der Convolutional Layer
	const int num_filters = 8; //Anzahl der Convolutionen pro Conv-Layer
	const int pool_layers_window = 2;
	const int pool_layers_stride = 2;
	const int conv_size1 = 3;
	const int conv_size2 = 3;
	const int num_weights = 10; //Anzahl an Klassifikations Klassen (10, da zehn Ziffern)
	const int imageSize;
	int packets = 12; //in wie viele arbeitspakete soll update aufgeteilt werden (falls parallel)
	int packetSizeConv; //groesse der arbeitspakete fuer update Conv
	int packetSizeFull; //groesse der arbeitspakete fuer update FullyConnectedLayer
	bool needCleanup; //soll JobCleanup aufgerufen werden?

	float corr1 = 1.0f;		//Korrekturterm des erstes Moments
	float corr2 = 1.0f;		//Korrekturterm des zweiten Moments

	int step; //Anzahl der bisher gelernten Batches

	vector<Conv> conv_layers; //Vektor aller Convolutional Layers
	vector<MaxPool> pooling_layers; //Vektor aller Pooling Layers
	FullyConnectedLayer *connected_layer; //Das eine Fully Connected layer

	vector<vector<vector<vector<float>>>> first_momentum_filters; //Erstes Moment der Filter: index1->Layer, index2->Filter des Layers index1, index3&4-> x bzw y Richtung des Filters
	vector<vector<vector<vector<float>>>> second_momentum_filters; //Zweites Moment der Filter: index1->Layer, index2->Filter des Layers index1, index3&4-> x bzw y Richtung des Filters
	vector<vector<float>> first_momentum_conv_biases; //Erstes Moment der Filterbiasse: index1->Layer, index2->Filter des Layers index1
	vector<vector<float>> second_momentum_conv_biases; //Zweites Moment der Filterbiasse: index1->Layer, index2->Filter des Layers index1
	vector<vector<float>> first_momentum_weights; //Erstes Moment der Gewichte: index1->Klassifikationklasse, index2->Gewichte der Pixel f�r Klassifikationklasse index1
	vector<vector<float>> second_momentum_weights; //Zweites Moment der Gewichte: index1->Klassifikationklasse, index2->Gewichte der Pixel f�r Klassifikationklasse index1
	vector<float> first_momentum_conn_biases; //Erstes Moment der Gewichtbiasse: index1->Klassifikationklasse
	vector<float> second_momentum_conn_biases; //Zweites Moment der Gewichtbiasse: index1->Klassifikationklasse

	CNN(int imageSize);

	tuple<float, bool> forward(vector<vector<vector<float>>> &image, int_fast8_t label);

	tuple<float, int_fast8_t> learn(vector<vector<vector<float>> > &x_batch, vector<int_fast8_t> &y_batch);

	void updateJob(int packet);

	void updateJobCleanup(int packet);

	void update_par();

	void update();

};

#endif /* CONV_H_ */
