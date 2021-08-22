/*
 * MaxPool.h
 *
 *  Created on: 24.06.2021
 *      Author: Stefan
 */

#ifndef MAXPOOL_H_
#define MAXPOOL_H_

#include <vector>

using namespace std;

class MaxPool {
public:
	int num_of_inputs, input_size1, input_size2;
	int window, stride;
	int output_size1, output_size2;
	vector<vector<vector<float>>> *input = nullptr; //pointer auf den input (forward param) (kann theoretisch auch nur einmal gesetzt werden, da pointer danach immer gleich bleibt)
	vector<vector<tuple<float, float>>> inputCoordsOfOutputPixels; //Koordinaten (relativ zum Window) des ausgewählten Pixels pro featureMap und pro Window dieser FeatureMap (geordnet)
	vector<vector<tuple<float, float>>> previouslyUsedLossInputPixels; //Koordinaten (relativ zum Window) des zuvor beschriebenen Pixels des LossInputs pro featureMap und pro Window dieser FeatureMap (geordnet)
	//index1->featureMap, index2&3-> x und y der FeatureMap
	vector<vector<vector<float>>> output; //index1->(generierte) featureMap (num_of_inputs viele), index2&3-> x und y der FeatureMap
	vector<vector<vector<float>>> *loss_gradient = nullptr; //pointer auf den loss gradienten (backprop param) (kann theoretisch auch nur einmal gesetzt werden, da pointer danach immer gleich bleibt)
	//index1->featureMap (num_of_inputs viele), index2&3-> x und y der FeatureMap
	vector<vector<vector<float>>> loss_input; //index1->featureMap (num_of_inputs viele), index2&3-> x und y der FeatureMap
	int packets = 12; //in wie viele arbeitspakete sollen forward/backprop aufgeteilt werden (falls parallel)
	int packetSize; //groesse der arbeitspakete
	bool needCleanup; //soll JobCleanup aufgerufen werden?

	MaxPool(int w, int s, int n, int s1, int s2);

	void forwardJob(int packet);

	void forwardJobCleanup(int packet);

	void forward(vector<vector<vector<float>>> &inputP);

	void forward_par(vector<vector<vector<float>>> &inputP);

	void backpropJob(int packet);

	void backpropJobCleanup(int packet);

	void backprop(vector<vector<vector<float>>> &loss_gradientP);

	void backprop_par(vector<vector<vector<float>>> &loss_gradientP);
};

#endif /* MAXPOOL_H_ */
