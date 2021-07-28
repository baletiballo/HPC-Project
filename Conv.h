/*
 * Conv.h
 *
 *  Created on: 24.06.2021
 *      Author: Stefan
 */

#ifndef CONV_H_
#define CONV_H_

#include <vector>

using namespace std;

class Conv {
public:
	int num_filters; //wie viele filter?
	int num_of_inputs, input_size1, input_size2; //wie viele feature maps besitzt input, und wie gross sind diese?
	int conv_size1, conv_size2; //groesse der filter
	int num_windows1, num_windows2; //anzahl der fenster die von jedem filter abgedeckt werden
	vector<vector<vector<float>>> filters; //index1->filter, index2&3-> x und y des Filters index1
	vector<float> biases; //index1->filter
	vector<vector<vector<float>>> *input = nullptr; //pointer auf den input (forward param) (kann theoretisch auch nur einmal gesetzt werden, da pointer danach immer gleich bleibt)
	//index1->featureMap, index2&3-> x und y der FeatureMap
	vector<vector<vector<float>>> output; //index1->(generierte) featureMap (num_of_inputs*num_filters viele), index2&3-> x und y der FeatureMap
	vector<vector<vector<float>>> *loss_gradient = nullptr; //pointer auf den loss gradienten (backprop param) (kann theoretisch auch nur einmal gesetzt werden, da pointer danach immer gleich bleibt)
	//index1->featureMap (num_of_inputs*num_filters viele, da wir ja die losses der generierten featureMaps betrachten), index2&3-> x und y der FeatureMap
	vector<vector<vector<float>>> filter_gradient; //index1->filter, index2&3-> x und y des Filters index1
	vector<float> bias_gradient; //index1->filter
	vector<vector<vector<float>>> loss_input; //index1->featureMap (num_of_inputs viele), index2&3-> x und y der FeatureMap
	int packets = 12; //in wie viele arbeitspakete sollen forward/backprop aufgeteilt werden (falls parallel)
	int packetSizeForw; //groesse der arbeitspakete fuer forward
	int packetSizeBack; //groesse der arbeitspakete fuer backprop
	bool needForwCleanup; //soll forwardJobCleanup aufgerufen werden?
	bool needBackCleanup; //soll backpropJobCleanup aufgerufen werden?

	Conv(int f, int c1, int c2, int n, int s1, int s2);

	void forwardJob(int packet);

	void forwardJobCleanup(int packet);

	void forward(vector<vector<vector<float>>> &inputP);

	void forward_par(vector<vector<vector<float>>> &inputP);

	void backpropJob(int packet);

	void backpropJobCleanup(int packet);

	void backprop(vector<vector<vector<float>>> &loss_gradientP);

	void backprop_par(vector<vector<vector<float>>> &loss_gradientP);

	void cleanup();
};

#endif /* CONV_H_ */
