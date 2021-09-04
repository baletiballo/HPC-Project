#include "FullyConnectedLayer.h"

FullyConnectedLayer::FullyConnectedLayer() {
	//mtx.resize(num_weights);

	weights  = new float [num_weights] [num_lastLayer_inputNeurons]{};
	biases = new float [num_weights]{};
	output = new float [num_weights]{};
	weight_gradient = new float [num_weights] [num_lastLayer_inputNeurons]{};
	bias_gradient = new float [num_weights]{};
	loss_input = new float [num_inputs] [input_size1] [input_size2]{};

	std::normal_distribution<float> distribution(0.0, 1.0);
	for (unsigned i = 0; i < num_weights; i++) 
	{
		for (unsigned j = 0; j < num_lastLayer_inputNeurons; j++) {
			std::random_device dev;
			std::default_random_engine generator(dev());
			weights[i][j] = distribution(generator) / (num_lastLayer_inputNeurons);
		}
	}
}

/**
 * Forward
 *
 * @param inputP
 */
void FullyConnectedLayer::forward(float inputP [num_filters] [input_size1] [input_size2]) {
	input = inputP;
	for (unsigned currClass = 0; currClass < num_weights; currClass++) {
		output[currClass] = biases[currClass];
		for (unsigned currFeatureMap = 0; currFeatureMap < num_inputs; currFeatureMap++) {
			for (unsigned k = 0; k < input_size1; k++) {
				//DEPENDS ON: input_size2
				//output[currClass] += input[currFeatureMap][k][l] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + l];
				output[currClass] += input[currFeatureMap][k][ 0] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 +  0];
				output[currClass] += input[currFeatureMap][k][ 1] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 +  1];
				output[currClass] += input[currFeatureMap][k][ 2] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 +  2];
				output[currClass] += input[currFeatureMap][k][ 3] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 +  3];
				output[currClass] += input[currFeatureMap][k][ 4] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 +  4];
				output[currClass] += input[currFeatureMap][k][ 5] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 +  5];
				output[currClass] += input[currFeatureMap][k][ 6] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 +  6];
				output[currClass] += input[currFeatureMap][k][ 7] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 +  7];
				output[currClass] += input[currFeatureMap][k][ 8] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 +  8];
				output[currClass] += input[currFeatureMap][k][ 9] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 +  9];
				output[currClass] += input[currFeatureMap][k][10] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 10];
				output[currClass] += input[currFeatureMap][k][11] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 11];
				output[currClass] += input[currFeatureMap][k][12] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 12];
				output[currClass] += input[currFeatureMap][k][13] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 13];
				output[currClass] += input[currFeatureMap][k][14] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 14];
				output[currClass] += input[currFeatureMap][k][15] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 15];
				output[currClass] += input[currFeatureMap][k][16] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 16];
				output[currClass] += input[currFeatureMap][k][17] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 17];
				output[currClass] += input[currFeatureMap][k][18] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 18];
				output[currClass] += input[currFeatureMap][k][19] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 19];
				output[currClass] += input[currFeatureMap][k][20] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 20];
				output[currClass] += input[currFeatureMap][k][21] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 21];
				output[currClass] += input[currFeatureMap][k][22] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 22];
				output[currClass] += input[currFeatureMap][k][23] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 23];
				output[currClass] += input[currFeatureMap][k][24] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 24];
				output[currClass] += input[currFeatureMap][k][25] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 25];
				output[currClass] += input[currFeatureMap][k][26] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 26];
				output[currClass] += input[currFeatureMap][k][27] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 27];
				output[currClass] += input[currFeatureMap][k][28] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 28];
				output[currClass] += input[currFeatureMap][k][29] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 29];
				output[currClass] += input[currFeatureMap][k][30] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 30];
				output[currClass] += input[currFeatureMap][k][31] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 31];
				output[currClass] += input[currFeatureMap][k][32] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 32];
				output[currClass] += input[currFeatureMap][k][33] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 33];
				output[currClass] += input[currFeatureMap][k][34] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 34];
				output[currClass] += input[currFeatureMap][k][35] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 35];
				output[currClass] += input[currFeatureMap][k][36] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 36];
				output[currClass] += input[currFeatureMap][k][37] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 37];
				output[currClass] += input[currFeatureMap][k][38] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 38];
				output[currClass] += input[currFeatureMap][k][39] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + 39];
				
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
		bias_gradient[cur_weight] = 0.0;
		for (unsigned i = 0; i < num_lastLayer_inputNeurons; i += 100) {
			weight_gradient[cur_weight][i] = 0.0;
			weight_gradient[cur_weight][i + 1]  = 0.0;
			weight_gradient[cur_weight][i + 2]  = 0.0;
			weight_gradient[cur_weight][i + 3]  = 0.0;
			weight_gradient[cur_weight][i + 4]  = 0.0;
			weight_gradient[cur_weight][i + 5]  = 0.0;
			weight_gradient[cur_weight][i + 6]  = 0.0;
			weight_gradient[cur_weight][i + 7]  = 0.0;
			weight_gradient[cur_weight][i + 8]  = 0.0;
			weight_gradient[cur_weight][i + 9]  = 0.0;
			weight_gradient[cur_weight][i + 10] = 0.0;
			weight_gradient[cur_weight][i + 11] = 0.0;
			weight_gradient[cur_weight][i + 12] = 0.0;
			weight_gradient[cur_weight][i + 13] = 0.0;
			weight_gradient[cur_weight][i + 14] = 0.0;
			weight_gradient[cur_weight][i + 15] = 0.0;
			weight_gradient[cur_weight][i + 16] = 0.0;
			weight_gradient[cur_weight][i + 17] = 0.0;
			weight_gradient[cur_weight][i + 18] = 0.0;
			weight_gradient[cur_weight][i + 19] = 0.0;
			weight_gradient[cur_weight][i + 20] = 0.0;
			weight_gradient[cur_weight][i + 21] = 0.0;
			weight_gradient[cur_weight][i + 22] = 0.0;
			weight_gradient[cur_weight][i + 23] = 0.0;
			weight_gradient[cur_weight][i + 24] = 0.0;
			weight_gradient[cur_weight][i + 25] = 0.0;
			weight_gradient[cur_weight][i + 26] = 0.0;
			weight_gradient[cur_weight][i + 27] = 0.0;
			weight_gradient[cur_weight][i + 28] = 0.0;
			weight_gradient[cur_weight][i + 29] = 0.0;
			weight_gradient[cur_weight][i + 30] = 0.0;
			weight_gradient[cur_weight][i + 31] = 0.0;
			weight_gradient[cur_weight][i + 32] = 0.0;
			weight_gradient[cur_weight][i + 33] = 0.0;
			weight_gradient[cur_weight][i + 34] = 0.0;
			weight_gradient[cur_weight][i + 35] = 0.0;
			weight_gradient[cur_weight][i + 36] = 0.0;
			weight_gradient[cur_weight][i + 37] = 0.0;
			weight_gradient[cur_weight][i + 38] = 0.0;
			weight_gradient[cur_weight][i + 39] = 0.0;
			weight_gradient[cur_weight][i + 40] = 0.0;
			weight_gradient[cur_weight][i + 41] = 0.0;
			weight_gradient[cur_weight][i + 42] = 0.0;
			weight_gradient[cur_weight][i + 43] = 0.0;
			weight_gradient[cur_weight][i + 44] = 0.0;
			weight_gradient[cur_weight][i + 45] = 0.0;
			weight_gradient[cur_weight][i + 46] = 0.0;
			weight_gradient[cur_weight][i + 47] = 0.0;
			weight_gradient[cur_weight][i + 48] = 0.0;
			weight_gradient[cur_weight][i + 49] = 0.0;
			weight_gradient[cur_weight][i + 50] = 0.0;
			weight_gradient[cur_weight][i + 51] = 0.0;
			weight_gradient[cur_weight][i + 52] = 0.0;
			weight_gradient[cur_weight][i + 53] = 0.0;
			weight_gradient[cur_weight][i + 54] = 0.0;
			weight_gradient[cur_weight][i + 55] = 0.0;
			weight_gradient[cur_weight][i + 56] = 0.0;
			weight_gradient[cur_weight][i + 57] = 0.0;
			weight_gradient[cur_weight][i + 58] = 0.0;
			weight_gradient[cur_weight][i + 59] = 0.0;
			weight_gradient[cur_weight][i + 60] = 0.0;
			weight_gradient[cur_weight][i + 61] = 0.0;
			weight_gradient[cur_weight][i + 62] = 0.0;
			weight_gradient[cur_weight][i + 63] = 0.0;
			weight_gradient[cur_weight][i + 64] = 0.0;
			weight_gradient[cur_weight][i + 65] = 0.0;
			weight_gradient[cur_weight][i + 66] = 0.0;
			weight_gradient[cur_weight][i + 67] = 0.0;
			weight_gradient[cur_weight][i + 68] = 0.0;
			weight_gradient[cur_weight][i + 69] = 0.0;
			weight_gradient[cur_weight][i + 70] = 0.0;
			weight_gradient[cur_weight][i + 71] = 0.0;
			weight_gradient[cur_weight][i + 72] = 0.0;
			weight_gradient[cur_weight][i + 73] = 0.0;
			weight_gradient[cur_weight][i + 74] = 0.0;
			weight_gradient[cur_weight][i + 75] = 0.0;
			weight_gradient[cur_weight][i + 76] = 0.0;
			weight_gradient[cur_weight][i + 77] = 0.0;
			weight_gradient[cur_weight][i + 78] = 0.0;
			weight_gradient[cur_weight][i + 79] = 0.0;
			weight_gradient[cur_weight][i + 80] = 0.0;
			weight_gradient[cur_weight][i + 81] = 0.0;
			weight_gradient[cur_weight][i + 82] = 0.0;
			weight_gradient[cur_weight][i + 83] = 0.0;
			weight_gradient[cur_weight][i + 84] = 0.0;
			weight_gradient[cur_weight][i + 85] = 0.0;
			weight_gradient[cur_weight][i + 86] = 0.0;
			weight_gradient[cur_weight][i + 87] = 0.0;
			weight_gradient[cur_weight][i + 88] = 0.0;
			weight_gradient[cur_weight][i + 89] = 0.0;
			weight_gradient[cur_weight][i + 90] = 0.0;
			weight_gradient[cur_weight][i + 91] = 0.0;
			weight_gradient[cur_weight][i + 92] = 0.0;
			weight_gradient[cur_weight][i + 93] = 0.0;
			weight_gradient[cur_weight][i + 94] = 0.0;
			weight_gradient[cur_weight][i + 95] = 0.0;
			weight_gradient[cur_weight][i + 96] = 0.0;
			weight_gradient[cur_weight][i + 97] = 0.0;
			weight_gradient[cur_weight][i + 98] = 0.0;
			weight_gradient[cur_weight][i + 99] = 0.0;
		}
	}
}

/**
 * Backprop (notice: gradients are getting already added here, since weight_gradient and bias_gradient only get reset to zero after a cleanup() call and are only needed after a batch)
 *
 * @param loss_gradientP
 */
void FullyConnectedLayer::backprop(float loss_gradientP [num_weights]) {
	loss_gradient = loss_gradientP;

	for (unsigned currFeatureMap = 0; currFeatureMap < num_inputs; currFeatureMap++) {
		for (unsigned currX = 0; currX < input_size1; currX++) {
			for (unsigned currY = 0; currY < input_size2; currY++) {
				//zero the loss Input, since the same method to just add them all together cannot be applied here
				loss_input[currFeatureMap][currX][currY] = 0.0;

				//DEPENDS ON: num_weights
				//loss_input[currFeatureMap][currX][currY] += weights[currClass][currFeatureMap * input_size1 * input_size2 + currX * input_size2 + currY] * loss_gradient[currClass];
				//weight_gradient[currClass][currFeatureMap * input_size1 * input_size2 + currX * input_size2 + currY] += loss_gradient[currClass] * input[currFeatureMap][currX][currY];
				loss_input[currFeatureMap][currX][currY] += weights[0][currFeatureMap * input_size1 * input_size2 + currX * input_size2 + currY] * loss_gradient[0];
				weight_gradient[0][currFeatureMap * input_size1 * input_size2 + currX * input_size2 + currY] += loss_gradient[0] * input[currFeatureMap][currX][currY];

				loss_input[currFeatureMap][currX][currY] += weights[1][currFeatureMap * input_size1 * input_size2 + currX * input_size2 + currY] * loss_gradient[1];
				weight_gradient[1][currFeatureMap * input_size1 * input_size2 + currX * input_size2 + currY] += loss_gradient[1] * input[currFeatureMap][currX][currY];

				loss_input[currFeatureMap][currX][currY] += weights[2][currFeatureMap * input_size1 * input_size2 + currX * input_size2 + currY] * loss_gradient[2];
				weight_gradient[2][currFeatureMap * input_size1 * input_size2 + currX * input_size2 + currY] += loss_gradient[2] * input[currFeatureMap][currX][currY];

				loss_input[currFeatureMap][currX][currY] += weights[3][currFeatureMap * input_size1 * input_size2 + currX * input_size2 + currY] * loss_gradient[3];
				weight_gradient[3][currFeatureMap * input_size1 * input_size2 + currX * input_size2 + currY] += loss_gradient[3] * input[currFeatureMap][currX][currY];
				
				loss_input[currFeatureMap][currX][currY] += weights[4][currFeatureMap * input_size1 * input_size2 + currX * input_size2 + currY] * loss_gradient[4];
				weight_gradient[4][currFeatureMap * input_size1 * input_size2 + currX * input_size2 + currY] += loss_gradient[4] * input[currFeatureMap][currX][currY];
				
				loss_input[currFeatureMap][currX][currY] += weights[5][currFeatureMap * input_size1 * input_size2 + currX * input_size2 + currY] * loss_gradient[5];
				weight_gradient[5][currFeatureMap * input_size1 * input_size2 + currX * input_size2 + currY] += loss_gradient[5] * input[currFeatureMap][currX][currY];
				
				loss_input[currFeatureMap][currX][currY] += weights[6][currFeatureMap * input_size1 * input_size2 + currX * input_size2 + currY] * loss_gradient[6];
				weight_gradient[6][currFeatureMap * input_size1 * input_size2 + currX * input_size2 + currY] += loss_gradient[6] * input[currFeatureMap][currX][currY];
				
				loss_input[currFeatureMap][currX][currY] += weights[7][currFeatureMap * input_size1 * input_size2 + currX * input_size2 + currY] * loss_gradient[7];
				weight_gradient[7][currFeatureMap * input_size1 * input_size2 + currX * input_size2 + currY] += loss_gradient[7] * input[currFeatureMap][currX][currY];
				
				loss_input[currFeatureMap][currX][currY] += weights[8][currFeatureMap * input_size1 * input_size2 + currX * input_size2 + currY] * loss_gradient[8];
				weight_gradient[8][currFeatureMap * input_size1 * input_size2 + currX * input_size2 + currY] += loss_gradient[8] * input[currFeatureMap][currX][currY];
				
				loss_input[currFeatureMap][currX][currY] += weights[9][currFeatureMap * input_size1 * input_size2 + currX * input_size2 + currY] * loss_gradient[9];
				weight_gradient[9][currFeatureMap * input_size1 * input_size2 + currX * input_size2 + currY] += loss_gradient[9] * input[currFeatureMap][currX][currY];
			}
		}
	}

	for (unsigned i = 0; i < num_weights; i++) {
		bias_gradient[i] += loss_gradient[i];
	}

}

///**
// * Description
// *
// * @param packet
// * @return
// */
//void FullyConnectedLayer::forwardJob(int packet) {
//	float tmp = 0.0;
//
//	for (int index = packet * packetSize; index < (packet + 1) * packetSize; index++) {
//		int currClass = index / (num_lastLayer_inputNeurons);
//		int weightIndex = index % num_lastLayer_inputNeurons;
//		int currFeatureMap = weightIndex / (input_size1 * input_size2);
//		int k = (weightIndex / input_size2) % input_size1;
//		int l = weightIndex % input_size2;
//
//		tmp += input[currFeatureMap][k][l] * weights[currClass][weightIndex];
//
//		if ((unsigned) weightIndex + 1 == num_lastLayer_inputNeurons) {
//			mtx[currClass].lock();
//			output[currClass] += tmp;
//			mtx[currClass].unlock();
//
//			tmp = 0.0;
//		}
//	}
//	if (tmp != 0.0) {
//		int currClass = (((packet + 1) * packetSize) - 1) / (num_lastLayer_inputNeurons);
//		mtx[currClass].lock();
//		output[currClass] += tmp;
//		mtx[currClass].unlock();
//	}
//	sem.V(1);
//}
//
///**
// * Description
// *
// * @param packet
// * @return
// */
//void FullyConnectedLayer::forwardJobCleanup(int packet) {
//	float tmp = 0.0;
//	for (int index = packet * packetSize; (unsigned) index < num_weights * num_lastLayer_inputNeurons; index++) {
//		int currClass = index / (num_lastLayer_inputNeurons);
//		int weightIndex = index % num_lastLayer_inputNeurons;
//		int currFeatureMap = weightIndex / (input_size1 * input_size2);
//		int k = (weightIndex / input_size2) % input_size1;
//		int l = weightIndex % input_size2;
//
//		tmp += input[currFeatureMap][k][l] * weights[currClass][weightIndex];
//
//		if ((unsigned) weightIndex + 1 == num_lastLayer_inputNeurons) {
//			mtx[currClass].lock();
//			output[currClass] += tmp;
//			mtx[currClass].unlock();
//
//			tmp = 0.0;
//		}
//	}
//	if (tmp != 0.0) {
//		int currClass = (((packet + 1) * packetSize) - 1) / (num_lastLayer_inputNeurons);
//		mtx[currClass].lock();
//		output[currClass] += tmp;
//		mtx[currClass].unlock();
//	}
//}
//
///**
// * Description
// *
// * @param input
// * @return
// */
//void FullyConnectedLayer::forward_par(float inputP [num_filters] [input_size1] [input_size2]) {
//	input = inputP;
//
//	sem.set(0);
//	pool.setFullyConnectedLayer(*this);
//	pool.setTask(5);
//	for (int i = 0; i < num_packets; i++)
//		pushJob(i);
//
//	if (needCleanup)
//		forwardJobCleanup(num_packets + 1);
//
//	sem.P(num_packets);
//
//	for (unsigned i = 0; i < num_weights; i++)
//		output[i] += biases[i];
//}
//
///**
// * Description
// *
// * @param packet
// * @return
// */
//void FullyConnectedLayer::backpropJob(int packet) {
//	for (int index = packet * packetSize; index < (packet + 1) * packetSize; index++) {
//		int currClass = index / (num_lastLayer_inputNeurons);
//		int weightIndex = index % num_lastLayer_inputNeurons;
//		int currFeatureMap = weightIndex / (input_size1 * input_size2);
//		int currX = (weightIndex / input_size2) % input_size1;
//		int currY = weightIndex % input_size2;
//
//		weight_gradient[currClass][weightIndex] = loss_gradient[currClass] * input[currFeatureMap][currX][currY];
//	}
//
//	for (unsigned weightIndex = packet; weightIndex < num_lastLayer_inputNeurons; weightIndex += num_packets) {
//		int currFeatureMap = weightIndex / (input_size1 * input_size2);
//		int currX = (weightIndex / input_size2) % input_size1;
//		int currY = weightIndex % input_size2;
//		loss_input[currFeatureMap][currX][currY] = 0.0;
//
//		//DEPENDS ON: num_weights
//		//loss_input[currFeatureMap][currX][currY] += weights[currClass][weightIndex] * loss_gradient[currClass];
//		loss_input[currFeatureMap][currX][currY] += weights[0][weightIndex] * loss_gradient[0];
//		loss_input[currFeatureMap][currX][currY] += weights[1][weightIndex] * loss_gradient[1];
//		loss_input[currFeatureMap][currX][currY] += weights[2][weightIndex] * loss_gradient[2];
//		loss_input[currFeatureMap][currX][currY] += weights[3][weightIndex] * loss_gradient[3];
//		loss_input[currFeatureMap][currX][currY] += weights[4][weightIndex] * loss_gradient[4];
//		loss_input[currFeatureMap][currX][currY] += weights[5][weightIndex] * loss_gradient[5];
//		loss_input[currFeatureMap][currX][currY] += weights[6][weightIndex] * loss_gradient[6];
//		loss_input[currFeatureMap][currX][currY] += weights[7][weightIndex] * loss_gradient[7];
//		loss_input[currFeatureMap][currX][currY] += weights[8][weightIndex] * loss_gradient[8];
//		loss_input[currFeatureMap][currX][currY] += weights[9][weightIndex] * loss_gradient[9];
//	}
//	sem.V(1);
//}
//
///**
// * Description
// *
// * @param packet
// * @return
// */
//void FullyConnectedLayer::backpropJobCleanup(int packet) {
//	for (int index = packet * packetSize; (unsigned) index < num_weights * num_lastLayer_inputNeurons; index++) {
//		int currClass = index / (num_lastLayer_inputNeurons);
//		int weightIndex = index % num_lastLayer_inputNeurons;
//		int currFeatureMap = weightIndex / (input_size1 * input_size2);
//		int currX = (weightIndex / input_size2) % input_size1;
//		int currY = weightIndex % input_size2;
//
//		weight_gradient[currClass][weightIndex] = loss_gradient[currClass] * input[currFeatureMap][currX][currY];
//	}
//
//	for (unsigned weightIndex = packet; weightIndex < num_lastLayer_inputNeurons; weightIndex += num_packets) {
//		int currFeatureMap = weightIndex / (input_size1 * input_size2);
//		int currX = (weightIndex / input_size2) % input_size1;
//		int currY = weightIndex % input_size2;
//		loss_input[currFeatureMap][currX][currY] = 0.0;
//
//		//DEPENDS ON: num_weights
//		//loss_input[currFeatureMap][currX][currY] += weights[currClass][weightIndex] * loss_gradient[currClass];
//		loss_input[currFeatureMap][currX][currY] += weights[0][weightIndex] * loss_gradient[0];
//		loss_input[currFeatureMap][currX][currY] += weights[1][weightIndex] * loss_gradient[1];
//		loss_input[currFeatureMap][currX][currY] += weights[2][weightIndex] * loss_gradient[2];
//		loss_input[currFeatureMap][currX][currY] += weights[3][weightIndex] * loss_gradient[3];
//		loss_input[currFeatureMap][currX][currY] += weights[4][weightIndex] * loss_gradient[4];
//		loss_input[currFeatureMap][currX][currY] += weights[5][weightIndex] * loss_gradient[5];
//		loss_input[currFeatureMap][currX][currY] += weights[6][weightIndex] * loss_gradient[6];
//		loss_input[currFeatureMap][currX][currY] += weights[7][weightIndex] * loss_gradient[7];
//		loss_input[currFeatureMap][currX][currY] += weights[8][weightIndex] * loss_gradient[8];
//		loss_input[currFeatureMap][currX][currY] += weights[9][weightIndex] * loss_gradient[9];
//	}
//}
//
///**
// * Description
// *
// * @param loss_gradientP
// * @return
// */
//void FullyConnectedLayer::backprop_par(float loss_gradientP [num_weights]) {
//	loss_gradient = loss_gradientP;
//
//	sem.set(0);
//	pool.setFullyConnectedLayer(*this);
//	pool.setTask(6);
//	for (int i = 0; i < num_packets; i++)
//		pushJob(i);
//
//	if (needCleanup)
//		backpropJobCleanup(num_packets + 1);
//
//	for (unsigned currClass = 0; currClass < num_weights; currClass++) //fast enough as is
//		bias_gradient[currClass] = loss_gradient[currClass];
//
//	sem.P(num_packets);
//}
