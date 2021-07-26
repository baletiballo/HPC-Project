/*
 * FullyConnectedLayer.h
 *
 *  Created on: 24.06.2021
 *      Author: Stefan
 */

#ifndef FULLYCONNECTEDLAYER_H_
#define FULLYCONNECTEDLAYER_H_

#include <vector>
#include <mutex>
#include <queue>

using namespace std;

class FullyConnectedLayer {
public:
	unsigned num_of_inputs, input_size1, input_size2;
	unsigned num_weights; //Anzahl der Klassifikationklassen
	unsigned total_size;  //Anzahl der Eingabe "Neuronen"
	vector<vector<float>> weights; //index1->Klassifikationklasse, index2->gewicht der Klassifikationklasse index1
	vector<float> biases; //index1->Klassifikationklasse
	vector<vector<vector<float>>> *input = nullptr; //pointer auf den input (forward param) (kann theoretisch auch nur einmal gesetzt werden, da pointer danach immer gleich bleibt)
	//index1->featureMap, index2&3-> x und y der FeatureMap
	vector<float> output; //index1->Klassifikationklasse
	vector<float> *loss_gradient = nullptr; //pointer auf den loss gradienten (backprop param) (kann theoretisch auch nur einmal gesetzt werden, da pointer danach immer gleich bleibt)
	//index1->Klassifikationklasse, index2->gewicht der Klassifikationklasse index1
	vector<vector<float>> weight_gradient; //index1->Klassifikationklasse, index2->gewicht der Klassifikationklasse index1
	vector<float> bias_gradient; //index1->Klassifikationklasse
	vector<vector<vector<float>>> loss_input; //index1->featureMap (num_of_inputs viele), index2&3-> x und y der FeatureMap
	int packets = 12; //in wie viele arbeitspakete sollen forward/backprop aufgeteilt werden (falls parallel)
	int packetSize; //groesse der arbeitspakete
	deque<mutex> mtx; //benoetigt fuer einige parallele aufteilungen, da ueberschneidungen von indizes der arbeitspakete passieren koennen

	FullyConnectedLayer(unsigned w, unsigned n, unsigned s1, unsigned s2);

	void forwardJob(int packet);

	void forwardJobCleanup(int packet);

	void forward(vector<vector<vector<float>>> &inputP);

	void forward_par(vector<vector<vector<float>>> &inputP);

	void backpropJob(int packet);

	void backpropJobCleanup(int packet);

	void backprop(vector<float> &loss_gradientP);

	void backprop_par(vector<float> &loss_gradientP);

	void cleanup();
};



#endif /* FULLYCONNECTEDLAYER_H_ */
