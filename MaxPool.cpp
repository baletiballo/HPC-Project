#include "MaxPool.h"
#include "ParallelStuff.h"

using namespace std;

MaxPool::MaxPool(int w, int s, int n, int s1, int s2) {
	window = w;
	stride = s;
	num_of_inputs = n;
	input_size1 = s1;
	input_size2 = s2;
	output_size1 = (input_size1 - window) / stride + 1;
	output_size2 = (input_size2 - window) / stride + 1;
	packetSize = (num_of_inputs * output_size1 * output_size2) / packets;
	needCleanup = (num_of_inputs * output_size1 * output_size2) % packets != 0;

	output.resize(num_of_inputs, vector<vector<float>>(output_size1, vector<float>(output_size2, 0.0)));

	loss_input.resize(num_of_inputs, vector<vector<float>>(input_size1, vector<float>(input_size2, 0.0)));

	inputCoordsOfOutputPixels.resize(num_of_inputs, vector<tuple<float, float>>(output_size1 * output_size2, tuple<float, float>(-1.0, -1.0)));
	previouslyUsedLossInputPixels.resize(num_of_inputs, vector<tuple<float, float>>(output_size1 * output_size2, tuple<float, float>(0.0, 0.0)));
}

/**
 * Forward
 *
 * @param inputP
 * @return
 */
void MaxPool::forward(vector<vector<vector<float>>> &inputP) {
	input = &inputP;

	for (int cur_featureMap = 0; cur_featureMap < num_of_inputs; cur_featureMap++) { //per input
		for (int i = 0; i < input_size1 - window; i += stride) {
			//per region
			for (int j = 0; j < input_size2 - window; j += stride) {
				// per region

				//matrix max pooling
				float max = (*input)[cur_featureMap][i][j];
				int maxX = 0;
				int maxY = 0;
				for (int m = 0; m < window; m++) {
					for (int n = 0; n < window; n++) {
						if (max < (*input)[cur_featureMap][i + m][j + n]) {
							max = (*input)[cur_featureMap][i + m][j + n];
							maxX = m;
							maxY = n;
						}
					}
				}

				output[cur_featureMap][i / stride][j / stride] = max;

				//Koordinaten schreiben
				get<0>(inputCoordsOfOutputPixels[cur_featureMap][i / stride + j / stride * output_size1]) = maxX;
				get<1>(inputCoordsOfOutputPixels[cur_featureMap][i / stride + j / stride * output_size1]) = maxY;
			}
		}
	}
}

/**
 * Backprop
 *
 * @param loss_gradientP
 * @return
 */
void MaxPool::backprop(vector<vector<vector<float>>> &loss_gradientP) {
	loss_gradient = &loss_gradientP;

	for (int cur_featureMap = 0; cur_featureMap < num_of_inputs; cur_featureMap++) { //per input
		for (int i = 0; i < input_size1 - window; i += stride) {
			//per region
			for (int j = 0; j < input_size2 - window; j += stride) {
				// per region

				//zero the loss Input
				int previousIndexX = get<0>(previouslyUsedLossInputPixels[cur_featureMap][i / stride + j / stride * output_size1]);
				int previousIndexY = get<1>(previouslyUsedLossInputPixels[cur_featureMap][i / stride + j / stride * output_size1]);
				loss_input[cur_featureMap][i + previousIndexX][j + previousIndexY] = 0.0;

				//matrix max pooling
				int indexX = get<0>(inputCoordsOfOutputPixels[cur_featureMap][i / stride + j / stride * output_size1]);
				int indexY = get<1>(inputCoordsOfOutputPixels[cur_featureMap][i / stride + j / stride * output_size1]);

				//set only the lossInput of the "pixel" max pool kept
				loss_input[cur_featureMap][i + indexX][j + indexY] = (*loss_gradient)[cur_featureMap][i / stride][j / stride];

				get<0>(previouslyUsedLossInputPixels[cur_featureMap][i / stride + j / stride * output_size1]) = indexX;
				get<1>(previouslyUsedLossInputPixels[cur_featureMap][i / stride + j / stride * output_size1]) = indexY;
			}
		}
	}
}

/**
 * Description
 *
 * @param packet
 * @return
 */
void MaxPool::forwardJob(int packet) {
	for (int index = packet * packetSize; index < (packet + 1) * packetSize; index++) {
		int cur_featureMap = index / (output_size1 * output_size2);
		int indexX = (index / output_size2) % output_size1 * stride;
		int indexY = index % output_size2 * stride;

		//matrix max pooling
		float max = (*input)[cur_featureMap][indexX][indexY];
		int maxX = 0;
		int maxY = 0;
		for (int m = 0; m < window; m++) {
			for (int n = 0; n < window; n++) {
				if (max < (*input)[cur_featureMap][indexX + m][indexY + n]) {
					max = (*input)[cur_featureMap][indexX + m][indexY + n];
					maxX = m;
					maxY = n;
				}
			}
		}

		output[cur_featureMap][indexX / stride][indexY / stride] = max;

		//Koordinaten schreiben
		get<0>(inputCoordsOfOutputPixels[cur_featureMap][indexX / stride + indexY / stride * output_size1]) = maxX;
		get<1>(inputCoordsOfOutputPixels[cur_featureMap][indexX / stride + indexY / stride * output_size1]) = maxY;
	}
	sem.V(1);
}

/**
 * Description
 *
 * @param packet
 * @return
 */
void MaxPool::forwardJobCleanup(int packet) {
	for (int index = packet * packetSize; index < num_of_inputs * output_size1 * output_size2; index++) {
		int cur_featureMap = index / (output_size1 * output_size2);
		int indexX = (index / output_size2) % output_size1 * stride;
		int indexY = index % output_size2 * stride;

		float max = (*input)[cur_featureMap][indexX][indexY];
		int maxX = 0;
		int maxY = 0;
		for (int m = 0; m < window; m++) {
			for (int n = 0; n < window; n++) {
				if (max < (*input)[cur_featureMap][indexX + m][indexY + n]) {
					max = (*input)[cur_featureMap][indexX + m][indexY + n];
					maxX = m;
					maxY = n;
				}
			}
		}

		output[cur_featureMap][indexX / stride][indexY / stride] = max;

		//Koordinaten schreiben
		get<0>(inputCoordsOfOutputPixels[cur_featureMap][indexX / stride + indexY / stride * output_size1]) = maxX;
		get<1>(inputCoordsOfOutputPixels[cur_featureMap][indexX / stride + indexY / stride * output_size1]) = maxY;
	}
}

/**
 * Description
 *
 * @param input
 * @return
 */
void MaxPool::forward_par(vector<vector<vector<float>>> &inputP) {
	input = &inputP;

	sem.set(0);
	pool.setMaxPool(*this);
	pool.setTask(3);
	for (int i = 0; i < packets; i++) {
		pushJob(i);
	}
	if (needCleanup) {
		forwardJobCleanup(packets + 1);
	}
	sem.P(packets);
}

/**
 * Description
 *
 * @param packet
 * @return
 */
void MaxPool::backpropJob(int packet) {
	for (int index = packet * packetSize; index < (packet + 1) * packetSize; index++) {
		int cur_featureMap = index / (output_size1 * output_size2);
		int i = (index / output_size2) % output_size1 * stride;
		int j = index % output_size2 * stride;

		//zero the loss Input
		int previousIndexX = get<0>(previouslyUsedLossInputPixels[cur_featureMap][i / stride + j / stride * output_size1]);
		int previousIndexY = get<1>(previouslyUsedLossInputPixels[cur_featureMap][i / stride + j / stride * output_size1]);
		loss_input[cur_featureMap][i + previousIndexX][j + previousIndexY] = 0.0;

		//matrix max pooling
		int indexX = get<0>(inputCoordsOfOutputPixels[cur_featureMap][i / stride + j / stride * output_size1]);
		int indexY = get<1>(inputCoordsOfOutputPixels[cur_featureMap][i / stride + j / stride * output_size1]);

		//set only the lossInput of the "pixel" max pool kept
		loss_input[cur_featureMap][i + indexX][j + indexY] = (*loss_gradient)[cur_featureMap][i / stride][j / stride];

		get<0>(previouslyUsedLossInputPixels[cur_featureMap][i / stride + j / stride * output_size1]) = indexX;
		get<1>(previouslyUsedLossInputPixels[cur_featureMap][i / stride + j / stride * output_size1]) = indexY;
	}
	sem.V(1);
}

/**
 * Description
 *
 * @param packet
 * @return
 */
void MaxPool::backpropJobCleanup(int packet) {
	for (int index = packet * packetSize; index < num_of_inputs * output_size1 * output_size2; index++) {
		int cur_featureMap = index / (output_size1 * output_size2);
		int i = (index / output_size2) % output_size1 * stride;
		int j = index % output_size2 * stride;

		//zero the loss Input
		int previousIndexX = get<0>(previouslyUsedLossInputPixels[cur_featureMap][i / stride + j / stride * output_size1]);
		int previousIndexY = get<1>(previouslyUsedLossInputPixels[cur_featureMap][i / stride + j / stride * output_size1]);
		loss_input[cur_featureMap][i + previousIndexX][j + previousIndexY] = 0.0;

		//matrix max pooling
		int indexX = get<0>(inputCoordsOfOutputPixels[cur_featureMap][i / stride + j / stride * output_size1]);
		int indexY = get<1>(inputCoordsOfOutputPixels[cur_featureMap][i / stride + j / stride * output_size1]);

		//set only the lossInput of the "pixel" max pool kept
		loss_input[cur_featureMap][i + indexX][j + indexY] = (*loss_gradient)[cur_featureMap][i / stride][j / stride];

		get<0>(previouslyUsedLossInputPixels[cur_featureMap][i / stride + j / stride * output_size1]) = indexX;
		get<1>(previouslyUsedLossInputPixels[cur_featureMap][i / stride + j / stride * output_size1]) = indexY;
	}
}

/**
 * Description
 *
 * @param loss_gradient
 * @return
 */
void MaxPool::backprop_par(vector<vector<vector<float>>> &loss_gradientP) {
	loss_gradient = &loss_gradientP;

	sem.set(0);
	pool.setMaxPool(*this);
	pool.setTask(4);
	for (int i = 0; i < packets; i++) {
		pushJob(i);
	}
	if (needCleanup) {
		backpropJobCleanup(packets + 1);
	}
	sem.P(packets);
}
