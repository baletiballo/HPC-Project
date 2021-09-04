#include "MaxPool.h"

using namespace std;

MaxPool::MaxPool() {
	output = new float [num_filters] [output_size1] [output_size2]{};
	loss_input = new float [num_inputs] [input_size1] [input_size2]{};
}

/**
 * Forward
 *
 * @param inputP
 * @return
 */
void MaxPool::forward(float inputP [num_filters] [input_size1] [input_size2]) {
	input = inputP;

	for (int cur_featureMap = 0; cur_featureMap < num_inputs; cur_featureMap++) { //per input
		for (int i = 0; i < input_size1 - window; i += stride) {
			//per region
			for (int j = 0; j < input_size2 - window; j += stride) {
				// per region

				//matrix max pooling
				float max = input[cur_featureMap][i][j];
				int maxX = 0;
				int maxY = 0;

				//DEPENDS ON: pool_layers_window(m, n)
				// if (max < input[cur_featureMap][i + m][j + n]) {
				//		max = input[cur_featureMap][i + m][j + n];
				//		maxX = m;
				//		maxY = n;
				// }
				if (max < input[cur_featureMap][i][j]) {
					max = input[cur_featureMap][i][j];
					maxX = 0;
					maxY = 0;
				}

				if (max < input[cur_featureMap][i + 1][j]) {
					max = input[cur_featureMap][i + 1][j];
					maxX = 1;
					maxY = 0;
				}

				if (max < input[cur_featureMap][i][j + 1]) {
					max = input[cur_featureMap][i][j + 1];
					maxX = 0;
					maxY = 1;
				}

				if (max < input[cur_featureMap][i + 1][j + 1]) {
					max = input[cur_featureMap][i + 1][j + 1];
					maxX = 1;
					maxY = 1;
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
void MaxPool::backprop(float loss_gradientP [num_filters] [output_size1] [output_size2]) {
	loss_gradient = loss_gradientP;

	for (int cur_featureMap = 0; cur_featureMap < num_inputs; cur_featureMap++) { //per input
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
				loss_input[cur_featureMap][i + indexX][j + indexY] = loss_gradient[cur_featureMap][i / stride][j / stride];

				get<0>(previouslyUsedLossInputPixels[cur_featureMap][i / stride + j / stride * output_size1]) = indexX;
				get<1>(previouslyUsedLossInputPixels[cur_featureMap][i / stride + j / stride * output_size1]) = indexY;
			}
		}
	}
}

///**
// * Description
// *
// * @param packet
// * @return
// */
//void MaxPool::forwardJob(int packet) {
//	for (int index = packet * packetSize; index < (packet + 1) * packetSize; index++) {
//		int cur_featureMap = index / (output_size1 * output_size2);
//		int i = (index / output_size2) % output_size1 * stride;
//		int j = index % output_size2 * stride;
//
//		//matrix max pooling
//		float max = input[cur_featureMap][i][j];
//		int maxX = 0;
//		int maxY = 0;
//
//		//DEPENDS ON: pool_layers_window(m, n)
//		// if (max < input[cur_featureMap][i + m][j + n]) {
//		//		max = input[cur_featureMap][i + m][j + n];
//		//		maxX = m;
//		//		maxY = n;
//		// }
//		if (max < input[cur_featureMap][i][j]) {
//			max = input[cur_featureMap][i][j];
//			maxX = 0;
//			maxY = 0;
//		}
//
//		if (max < input[cur_featureMap][i + 1][j]) {
//			max = input[cur_featureMap][i + 1][j];
//			maxX = 1;
//			maxY = 0;
//		}
//
//		if (max < input[cur_featureMap][i][j + 1]) {
//			max = input[cur_featureMap][i][j + 1];
//			maxX = 0;
//			maxY = 1;
//		}
//
//		if (max < input[cur_featureMap][i + 1][j + 1]) {
//			max = input[cur_featureMap][i + 1][j + 1];
//			maxX = 1;
//			maxY = 1;
//		}
//
//		output[cur_featureMap][i / stride][j / stride] = max;
//
//		//Koordinaten schreiben
//		get<0>(inputCoordsOfOutputPixels[cur_featureMap][i / stride + j / stride * output_size1]) = maxX;
//		get<1>(inputCoordsOfOutputPixels[cur_featureMap][i / stride + j / stride * output_size1]) = maxY;
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
//void MaxPool::forwardJobCleanup(int packet) {
//	for (int index = packet * packetSize; index < num_inputs * output_size1 * output_size2; index++) {
//		int cur_featureMap = index / (output_size1 * output_size2);
//		int i = (index / output_size2) % output_size1 * stride;
//		int j = index % output_size2 * stride;
//
//		float max = input[cur_featureMap][i][j];
//		int maxX = 0;
//		int maxY = 0;
//
//		//DEPENDS ON: pool_layers_window(m, n)
//		// if (max < input[cur_featureMap][i + m][j + n]) {
//		//		max = input[cur_featureMap][i + m][j + n];
//		//		maxX = m;
//		//		maxY = n;
//		// }
//		if (max < input[cur_featureMap][i][j]) {
//			max = input[cur_featureMap][i][j];
//			maxX = 0;
//			maxY = 0;
//		}
//
//		if (max < input[cur_featureMap][i + 1][j]) {
//			max = input[cur_featureMap][i + 1][j];
//			maxX = 1;
//			maxY = 0;
//		}
//
//		if (max < input[cur_featureMap][i][j + 1]) {
//			max = input[cur_featureMap][i][j + 1];
//			maxX = 0;
//			maxY = 1;
//		}
//
//		if (max < input[cur_featureMap][i + 1][j + 1]) {
//			max = input[cur_featureMap][i + 1][j + 1];
//			maxX = 1;
//			maxY = 1;
//		}
//
//		output[cur_featureMap][i / stride][j / stride] = max;
//
//		//Koordinaten schreiben
//		get<0>(inputCoordsOfOutputPixels[cur_featureMap][i / stride + j / stride * output_size1]) = maxX;
//		get<1>(inputCoordsOfOutputPixels[cur_featureMap][i / stride + j / stride * output_size1]) = maxY;
//	}
//}
//
///**
// * Description
// *
// * @param input
// * @return
// */
//void MaxPool::forward_par(float inputP [num_filters] [input_size1] [input_size2]) {
//	input = inputP;
//
//	sem.set(0);
//	pool.setMaxPool(*this);
//	pool.setTask(3);
//	for (int i = 0; i < num_packets; i++)
//		pushJob(i);
//
//	if (needCleanup)
//		forwardJobCleanup(num_packets + 1);
//
//	sem.P(num_packets);
//}
//
///**
// * Description
// *
// * @param packet
// * @return
// */
//void MaxPool::backpropJob(int packet) {
//	for (int index = packet * packetSize; index < (packet + 1) * packetSize; index++) {
//		int cur_featureMap = index / (output_size1 * output_size2);
//		int i = (index / output_size2) % output_size1 * stride;
//		int j = index % output_size2 * stride;
//
//		//zero the loss Input
//		int previousIndexX = get<0>(previouslyUsedLossInputPixels[cur_featureMap][i / stride + j / stride * output_size1]);
//		int previousIndexY = get<1>(previouslyUsedLossInputPixels[cur_featureMap][i / stride + j / stride * output_size1]);
//		loss_input[cur_featureMap][i + previousIndexX][j + previousIndexY] = 0.0;
//
//		//matrix max pooling
//		int indexX = get<0>(inputCoordsOfOutputPixels[cur_featureMap][i / stride + j / stride * output_size1]);
//		int indexY = get<1>(inputCoordsOfOutputPixels[cur_featureMap][i / stride + j / stride * output_size1]);
//
//		//set only the lossInput of the "pixel" max pool kept
//		loss_input[cur_featureMap][i + indexX][j + indexY] = loss_gradient[cur_featureMap][i / stride][j / stride];
//
//		get<0>(previouslyUsedLossInputPixels[cur_featureMap][i / stride + j / stride * output_size1]) = indexX;
//		get<1>(previouslyUsedLossInputPixels[cur_featureMap][i / stride + j / stride * output_size1]) = indexY;
//	}
//
//	sem.V(1);
//}
//
///**
// * Description
// *
// * @param packet
// * @return
// */
//void MaxPool::backpropJobCleanup(int packet) {
//	for (int index = packet * packetSize; index < num_inputs * output_size1 * output_size2; index++) {
//		int cur_featureMap = index / (output_size1 * output_size2);
//		int i = (index / output_size2) % output_size1 * stride;
//		int j = index % output_size2 * stride;
//
//		//zero the loss Input
//		int previousIndexX = get<0>(previouslyUsedLossInputPixels[cur_featureMap][i / stride + j / stride * output_size1]);
//		int previousIndexY = get<1>(previouslyUsedLossInputPixels[cur_featureMap][i / stride + j / stride * output_size1]);
//		loss_input[cur_featureMap][i + previousIndexX][j + previousIndexY] = 0.0;
//
//		//matrix max pooling
//		int indexX = get<0>(inputCoordsOfOutputPixels[cur_featureMap][i / stride + j / stride * output_size1]);
//		int indexY = get<1>(inputCoordsOfOutputPixels[cur_featureMap][i / stride + j / stride * output_size1]);
//
//		//set only the lossInput of the "pixel" max pool kept
//		loss_input[cur_featureMap][i + indexX][j + indexY] = loss_gradient[cur_featureMap][i / stride][j / stride];
//
//		get<0>(previouslyUsedLossInputPixels[cur_featureMap][i / stride + j / stride * output_size1]) = indexX;
//		get<1>(previouslyUsedLossInputPixels[cur_featureMap][i / stride + j / stride * output_size1]) = indexY;
//	}
//}
//
///**
// * Description
// *
// * @param loss_gradient
// * @return
// */
//void MaxPool::backprop_par(float loss_gradientP [num_filters] [output_size1] [output_size2]) {
//	loss_gradient = loss_gradientP;
//
//	sem.set(0);
//	pool.setMaxPool(*this);
//	pool.setTask(4);
//	for (int i = 0; i < num_packets; i++)
//		pushJob(i);
//
//	if (needCleanup)
//		backpropJobCleanup(num_packets + 1);
//
//	sem.P(num_packets);
//}
