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
	vector<vector<float>> weights;
	vector<float> biases;
	vector<float> *curr_input = nullptr;
	vector<float> *curr_output = nullptr;
	vector<float> *curr_loss_gradient = nullptr;
	vector<vector<float>> *curr_weight_gradient = nullptr;
	vector<float> *curr_bias_gradient = nullptr;
	vector<float> *curr_loss_input = nullptr;
	int packets = 12;
	int packetSize;
	deque<mutex> mtx;
	deque<mutex> mtx_big;

	FullyConnectedLayer(unsigned w, unsigned n, unsigned s1, unsigned s2);

	void forwardJob(int packet);

	void forwardJobCleanup(int packet);

	vector<float> forward(vector<float> &input);

	void backpropJob(int packet);

	void backpropJobCleanup(int packet);

	tuple<vector<vector<float>>, vector<float>, vector<float>> backprop(vector<float> &loss_gradient, vector<float> &last_input);
};



#endif /* FULLYCONNECTEDLAYER_H_ */
