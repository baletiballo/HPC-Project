#include "FullyConnectedLayer.h"
#include <random>
#include "ParallelStuff.h"

FullyConnectedLayer::FullyConnectedLayer(unsigned w, unsigned n, unsigned s1, unsigned s2) {
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

	output.resize(num_weights);

	weight_gradient.resize(num_weights, vector<float>(total_size));
	bias_gradient.resize(num_weights);
	loss_input.resize(num_of_inputs, vector<vector<float>>(input_size1, vector<float>(input_size2, 0.0)));

	normal_distribution<float> distribution(0.0, 1.0);
	for (unsigned i = 0; i < num_weights; i++) {

		for (unsigned j = 0; j < total_size; j++) {
			random_device dev;
			default_random_engine generator(dev());
			weights[i][j] = distribution(generator) / (total_size);
		}
	}
}

/**
 * Forward
 *
 * @param inputP
 */
void FullyConnectedLayer::forward(vector<vector<vector<float>>> &inputP) {
	input = &inputP;
	for (unsigned i = 0; i < num_weights; i++) {
		output[i] = biases[i];
		for (unsigned j = 0; j < num_of_inputs; j++) {
			for (unsigned k = 0; k < input_size1; k++) {
				for (unsigned l = 0; l < input_size2; l++) {
					output[i] += (*input)[j][k][l] * weights[i][j * input_size1 * input_size2 + k * input_size2 + l];
				}
			}
		}
	}
}

/**
 * Call this after every Batch, addition of the gradients throughout a Batch is now done directly in backprop
 *
 */
void FullyConnectedLayer::cleanup() {
	for (unsigned cur_weight = 0; cur_weight < num_weights; cur_weight++) {
		bias_gradient[cur_weight] = 0;
		for (unsigned i = 0; i < total_size; i++) {
			weight_gradient[cur_weight][i] = 0;
		}
	}
}

/**
 * Backprop (notice: gradients are getting already added here, since weight_gradient and bias_gradient only get reset to zero after a cleanup() call and are only needed after a batch)
 *
 * @param loss_gradientP
 */
void FullyConnectedLayer::backprop(vector<float> &loss_gradientP) {
	loss_gradient = &loss_gradientP;

	for (unsigned i = 0; i < num_weights; i++) { //zero the loss Input, since the same method to just add them all together cannot be applied here
		for (unsigned j = 0; j < num_of_inputs; j++) {
			for (unsigned k = 0; k < input_size1; k++) {
				for (unsigned l = 0; l < input_size2; l++) {
					weight_gradient[i][j * input_size1 * input_size2 + k * input_size2 + l] += (*loss_gradient)[i] * (*input)[j][k][l];
				}
			}
		}
	}

	for (unsigned i = 0; i < num_weights; i++) {
		bias_gradient[i] += (*loss_gradient)[i];
	}

	for (unsigned i = 0; i < num_of_inputs; i++) {
		for (unsigned j = 0; j < input_size1; j++) {
			for (unsigned k = 0; k < input_size2; k++) {
				loss_input[i][j][k] = 0;
			}
		}
	}

	for (unsigned i = 0; i < num_of_inputs; i++) {
		for (unsigned j = 0; j < input_size1; j++) {
			for (unsigned k = 0; k < input_size2; k++) {
				for (unsigned l = 0; l < num_weights; l++) {
					loss_input[i][j][k] += weights[l][i * input_size1 * input_size2 + j * input_size2 + k] * (*loss_gradient)[l];
				}
			}
		}
	}

}

///**
// * Description
// *
// * @param packet
// * @return
//*/
//void FullyConnectedLayer::forwardJob(int packet) {
//	float tmp = 0.0;
//	for (int i = packet * packetSize; i < (packet + 1) * packetSize; i++) {
//		int index1 = i / (total_size);
//		int index2 = i % total_size;
//
//		tmp += (*curr_input)[index2] * weights[index1][index2];
//
//		if ((unsigned) index2 + 1 == total_size) {
//			mtx[index1].lock();
//			(*curr_output)[index1] += tmp;
//			mtx[index1].unlock();
//
//			tmp = 0.0;
//		}
//	}
//	if (tmp != 0.0) {
//		int index1 = (((packet + 1) * packetSize) - 1) / (total_size);
//		mtx[index1].lock();
//		(*curr_output)[index1] += tmp;
//		mtx[index1].unlock();
//	}
//	sem.V(1);
//}
//
// /**
// * Description
// *
// * @param packet
// * @return
//*/
//void FullyConnectedLayer::forwardJobCleanup(int packet) {
//	float tmp = 0.0;
//	for (int i = packet * packetSize; (unsigned) i < num_weights * total_size; i++) {
//		int index1 = i / (total_size);
//		int index2 = i % total_size;
//
//		tmp += (*curr_input)[index2] * weights[index1][index2];
//
//		if ((unsigned) index2 + 1 == total_size) {
//			mtx[index1].lock();
//			(*curr_output)[index1] += tmp;
//			mtx[index1].unlock();
//
//			tmp = 0.0;
//		}
//	}
//	if (tmp != 0.0) {
//		int index1 = (((packet + 1) * packetSize) - 1) / (total_size);
//		mtx[index1].lock();
//		(*curr_output)[index1] += tmp;
//		mtx[index1].unlock();
//	}
//}
//
// /**
// * Description
// *
// * @param input
// * @return
//*/
//vector<float> FullyConnectedLayer::forward(vector<float> &input) {
//	vector<float> output(num_weights);
//	curr_input = &input;
//	curr_output = &output;
//
//	sem.set(0);
//	pool.setFullyConnectedLayer(*this);
//			pool.setTask(5);
//	for (int i = 0; i < packets; i++) {
//		pushJob(i);
//	}
//	if ((num_weights * total_size) % packets != 0) {
//		forwardJobCleanup(packets + 1);
//	}
//	sem.P(packets);
//
//	for (unsigned i = 0; i < num_weights; i++) {
//		output[i] += biases[i];
//	}
//	return output;
//}
//
///**
// * Description
// *
// * @param packet
// * @return
//*/
//void FullyConnectedLayer::backpropJob(int packet) {
//	for (int i = packet * packetSize; i < (packet + 1) * packetSize; i++) {
//		int index1 = i / (total_size);
//		int index2 = i % total_size;
//
//		(*curr_weight_gradient)[index1][index2] = (*curr_loss_gradient)[index1] * (*curr_input)[index2];
//
//		mtx_big[index2].lock();
//		(*curr_loss_input)[index2] += weights[index1][index2] * (*curr_loss_gradient)[index1];
//		mtx_big[index2].unlock();
//	}
//	sem.V(1);
//}
//
//void FullyConnectedLayer::backpropJobCleanup(int packet) {
//	for (int i = packet * packetSize; (unsigned) i < num_weights * total_size; i++) {
//		int index1 = i / (total_size);
//		int index2 = i % total_size;
//
//		(*curr_weight_gradient)[index1][index2] = (*curr_loss_gradient)[index1] * (*curr_input)[index2];
//
//		mtx_big[index2].lock();
//		(*curr_loss_input)[index2] += weights[index1][index2] * (*curr_loss_gradient)[index1];
//		mtx_big[index2].unlock();
//	}
//}
//
// /**
// * Description
// *
// * @param loss_gradient
// * @param last_input
// * @return
//*/
//tuple<vector<vector<float>>, vector<float>, vector<float>> FullyConnectedLayer::backprop(vector<float> &loss_gradient, vector<float> &last_input) {
//	vector<vector<float>> weight_gradient(num_weights, vector<float>(total_size));
//	vector<float> bias_gradient(num_weights);
//	vector<float> loss_input(total_size, 0.0);
//
//	curr_weight_gradient = &weight_gradient;
//	curr_bias_gradient = &bias_gradient;
//	curr_loss_input = &loss_input;
//	curr_loss_gradient = &loss_gradient;
//	curr_input = &last_input;
//
//	sem.set(0);
//	pool.setFullyConnectedLayer(*this);
//		pool.setTask(6);
//	for (int i = 0; i < packets; i++) {
//		pushJob(i);
//	}
//	if ((num_weights * total_size) % packets != 0) {
//		backpropJobCleanup(packets + 1);
//	}
//	for (unsigned i = 0; i < num_weights; i++) { //fast enough as is
//		(*curr_bias_gradient)[i] = (*curr_loss_gradient)[i];
//	}
//	sem.P(packets);
//
//	return {weight_gradient, bias_gradient, loss_input};
//}
