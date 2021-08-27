/*
 * MaxPool.h
 *
 *  Created on: 24.06.2021
 *      Author: Stefan
 */

#ifndef MAXPOOL_H_
#define MAXPOOL_H_

#include <tuple>

#include "parameter.h"
#include "Conv.h"
#include "ParallelStuff.h"

using namespace std;

class MaxPool {

	static const int num_inputs		= num_filters;
	static const int input_size1	= imageSizeX_afterConvolution;
	static const int input_size2	= imageSizeY_afterConvolution;
	static const int stride 		= pool_layers_stride;
	static const int window 		= pool_layers_window;
	static const int output_size1	= imageSizeX_afterPooling;
	static const int output_size2	= imageSizeY_afterPooling;

public:

	float (*input) [input_size1] [input_size2]; //pointer auf den input (forward param) (kann theoretisch auch nur einmal gesetzt werden, da pointer danach immer gleich bleibt)
	tuple<float, float> inputCoordsOfOutputPixels [num_inputs] [output_size1 * output_size2]; //Koordinaten (relativ zum Window) des ausgew�hlten Pixels pro featureMap und pro Window dieser FeatureMap (geordnet)
	tuple<float, float> previouslyUsedLossInputPixels [num_inputs] [output_size1 * output_size2]; //Koordinaten (relativ zum Window) des zuvor beschriebenen Pixels des LossInputs pro featureMap und pro Window dieser FeatureMap (geordnet)
	//index1->featureMap, index2&3-> x und y der FeatureMap
	float output [num_filters] [output_size1] [output_size2]; //index1->(generierte) featureMap (num_inputs viele), index2&3-> x und y der FeatureMap
	float (*loss_gradient) [output_size1] [output_size2]; //pointer auf den loss gradienten (backprop param) (kann theoretisch auch nur einmal gesetzt werden, da pointer danach immer gleich bleibt)
	//index1->featureMap (num_inputs viele), index2&3-> x und y der FeatureMap
	float loss_input [num_inputs] [input_size1] [input_size2]; //index1->featureMap (num_inputs viele), index2&3-> x und y der FeatureMap
	int packetSize; //groesse der arbeitspakete
	bool needCleanup; //soll JobCleanup aufgerufen werden?

	MaxPool();

	void forwardJob(int packet);

	void forwardJobCleanup(int packet);

	void forward(float inputP [num_filters] [input_size1] [input_size2]);

	void forward_par(float inputP [num_filters] [input_size1] [input_size2]);

	void backpropJob(int packet);

	void backpropJobCleanup(int packet);

	void backprop(float loss_gradientP [num_filters] [output_size1] [output_size2]);

	void backprop_par(float loss_gradientP [num_filters] [output_size1] [output_size2]);
};

#endif /* MAXPOOL_H_ */
