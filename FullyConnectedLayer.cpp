#pragma once

#include <vector>
#include <random>
#include "ParallelStuff.cpp"

using namespace std;


class FullyConnectedLayer {
public:
	unsigned num_of_inputs, input_size1, input_size2;
	unsigned num_weights;
	unsigned total_size;
	vector<vector<float>> weights;
	vector<float> biases;
	vector<float> *curr_input = nullptr;
	vector<float> *curr_output = nullptr;
	vector<float> *curr_loss_gradient = nullptr;
	vector<vector<float>> *curr_weight_gradient = nullptr;
	vector<float> *curr_bias_gradient = nullptr;
	vector<float> *curr_loss_input = nullptr;
	int packets = 64;
	int packetSize;
	deque<mutex> mtx;
	deque<mutex> mtx_big;

	FullyConnectedLayer(unsigned w, unsigned n, unsigned s1, unsigned s2) {
		num_weights = w;
		num_of_inputs = n;
		input_size1 = s1;
		input_size2 = s2;
		total_size = num_of_inputs * input_size1 * input_size2;
		weights.resize(num_weights, vector<float>(total_size));
		biases.resize(num_weights, 0.0);
		packetSize = (num_weights * total_size) / packets;
		mtx.resize(num_weights);
		mtx_big.resize(total_size);

		normal_distribution<float> distribution(0.0, 1.0);
		for (unsigned i = 0; i < num_weights; i++) {

			for (unsigned j = 0; j < total_size; j++) {
				random_device dev;
				default_random_engine generator(dev());
				weights[i][j] = distribution(generator) / (total_size);
			}
		}
	}

	void forwardJob(int packet) {
		float tmp = 0.0;
		for (int i = packet * packetSize; i < (packet + 1) * packetSize; i++) {
			int index1 = i / (total_size);
			int index2 = i % total_size;

			tmp += (*curr_input)[index2] * weights[index1][index2];

			if ((unsigned) index2 + 1 == total_size) {
				mtx[index1].lock();
				(*curr_output)[index1] += tmp;
				mtx[index1].unlock();

				tmp = 0.0;
			}
		}
		if (tmp != 0.0) {
			int index1 = (((packet + 1) * packetSize) - 1) / (total_size);
			mtx[index1].lock();
			(*curr_output)[index1] += tmp;
			mtx[index1].unlock();
		}
		sem.V(1);
	}

	void forwardJobCleanup(int packet) {
		float tmp = 0.0;
		for (int i = packet * packetSize; (unsigned) i < num_weights * total_size; i++) {
			int index1 = i / (total_size);
			int index2 = i % total_size;

			tmp += (*curr_input)[index2] * weights[index1][index2];

			if ((unsigned) index2 + 1 == total_size) {
				mtx[index1].lock();
				(*curr_output)[index1] += tmp;
				mtx[index1].unlock();

				tmp = 0.0;
			}
		}
		if (tmp != 0.0) {
			int index1 = (((packet + 1) * packetSize) - 1) / (total_size);
			mtx[index1].lock();
			(*curr_output)[index1] += tmp;
			mtx[index1].unlock();
		}
	}

	vector<float> forward(vector<float> &input) {
		vector<float> output(num_weights);
		curr_input = &input;
		curr_output = &output;

		sem.set(0);
		for (int i = 0; i < packets; i++) {
			packaged_task<void()> job(bind(&forwardJob, this, i));
			pushJob(move(job));
		}
		if ((num_weights * total_size) % packets != 0) {
			forwardJobCleanup(packets + 1);
		}
		sem.P(packets);

		for (unsigned i = 0; i < num_weights; i++) {
			output[i] += biases[i];
		}
		return output;
	}

	void backpropJob(int packet) {
		for (int i = packet * packetSize; i < (packet + 1) * packetSize; i++) {
			int index1 = i / (total_size);
			int index2 = i % total_size;

			(*curr_weight_gradient)[index1][index2] = (*curr_loss_gradient)[index1] * (*curr_input)[index2];

			mtx_big[index2].lock();
			(*curr_loss_input)[index2] += weights[index1][index2] * (*curr_loss_gradient)[index1];
			mtx_big[index2].unlock();
		}
		sem.V(1);
	}

	void backpropJobCleanup(int packet) {
		for (int i = packet * packetSize; (unsigned) i < num_weights * total_size; i++) {
			int index1 = i / (total_size);
			int index2 = i % total_size;

			(*curr_weight_gradient)[index1][index2] = (*curr_loss_gradient)[index1] * (*curr_input)[index2];

			mtx_big[index2].lock();
			(*curr_loss_input)[index2] += weights[index1][index2] * (*curr_loss_gradient)[index1];
			mtx_big[index2].unlock();
		}
	}

	tuple<vector<vector<float>>, vector<float>, vector<float>> backprop(vector<float> &loss_gradient, vector<float> &last_input) {
		vector<vector<float>> weight_gradient(num_weights, vector<float>(total_size));
		vector<float> bias_gradient(num_weights);
		vector<float> loss_input(total_size, 0.0);

		curr_weight_gradient = &weight_gradient;
		curr_bias_gradient = &bias_gradient;
		curr_loss_input = &loss_input;
		curr_loss_gradient = &loss_gradient;
		curr_input = &last_input;

		sem.set(0);
		for (int i = 0; i < packets; i++) {
			packaged_task<void()> job(bind(&backpropJob, this, i));
			pushJob(move(job));
		}
		if ((num_weights * total_size) % packets != 0) {
			backpropJobCleanup(packets + 1);
		}
		for (unsigned i = 0; i < num_weights; i++) { //fast enough as is
			(*curr_bias_gradient)[i] = (*curr_loss_gradient)[i];
		}
		sem.P(packets);

		return {weight_gradient, bias_gradient, loss_input};
	}
};
