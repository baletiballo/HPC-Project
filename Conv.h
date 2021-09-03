/*
 * Conv.h
 *
 *  Created on: 24.06.2021
 *      Author: Stefan
 */

#ifndef CONV_H_
#define CONV_H_

#include <random>

#include "parameter.h"
#include "ParallelStuff.h"

class Conv {
	
	static const int num_windowsX = imageSizeX_afterConvolution; 
	static const int num_windowsY = imageSizeY_afterConvolution;
	static const int packetSizeForw = (num_filters * num_windowsX * num_windowsY) / num_packets; //groesse der arbeitspakete fuer forward
	static const int packetSizeBack = (num_windowsX * num_windowsY) / num_packets; //groesse der arbeitspakete fuer backprop
	static const bool needForwCleanup = (num_filters * num_windowsX * num_windowsY) % num_packets != 0; //soll forwardJobCleanup aufgerufen werden?
	static const bool needBackCleanup = (num_windowsX * num_windowsY) % num_packets != 0; //soll backpropJobCleanup aufgerufen werden?

public:
	
	float (*filters) [conv_size1] [conv_size2]; //index1->filter, index2&3-> x und y des Filters index1
	float (*biases); //index1->filter
	float (*output) [num_windowsX] [num_windowsY]; //index1->(generierte) featureMap (num_inputs*num_filters viele), index2&3-> x und y der FeatureMap
	float (*filter_gradient) [conv_size1] [conv_size2]; //index1->filter, index2&3-> x und y des Filters index1
	float (*bias_gradient); //index1->filter
	float (*loss_input) [imageSizeY]; //index1->featureMap (num_inputs viele), index2&3-> x und y der FeatureMap
	
	float (*input) [imageSizeY]; //pointer auf den input (forward param) (kann theoretisch auch nur einmal gesetzt werden, da pointer danach immer gleich bleibt)
	//index1->featureMap, index2&3-> x und y der FeatureMap
	float (*loss_gradient) [num_windowsX] [num_windowsY]; //pointer auf den loss gradienten (backprop param) (kann theoretisch auch nur einmal gesetzt werden, da pointer danach immer gleich bleibt)
	//index1->featureMap num_filters viele, da wir ja die losses der generierten featureMaps betrachten), index2&3-> x und y der FeatureMap

	Conv();

	void forwardJob(int packet);

	void forwardJobCleanup(int packet);

	void forward(float inputP [imageSizeX] [imageSizeY]);

	void forward_par(float inputP [imageSizeX] [imageSizeY]);

	void backpropJob(int packet);

	void backpropJobCleanup(int packet);

	void backprop(float loss_gradientP [num_filters] [num_windowsX] [num_windowsY]);

	void backprop_par(float loss_gradientP [num_filters] [num_windowsX] [num_windowsY]);

	void cleanup();
};

#endif /* CONV_H_ */
