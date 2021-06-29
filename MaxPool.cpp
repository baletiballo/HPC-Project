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

	output.resize(num_of_inputs, vector<vector<float>>(output_size1, vector<float>(output_size2, 0.0)));

	loss_input.resize(num_of_inputs, vector<vector<float>>(input_size1, vector<float>(input_size2, 0.0)));
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
				for (int m = 0; m < window; m++) {
					for (int n = 0; n < window; n++)
						if (max < (*input)[cur_featureMap][i + m][j + n])
							max = (*input)[cur_featureMap][i + m][j + n];

					output[cur_featureMap][i / stride][j / stride] = max;
				}
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

	for (int cur_featureMap = 0; cur_featureMap < num_of_inputs; cur_featureMap++) { //zero the loss Input
		for (int i = 0; i < input_size1; i++) {
			for (int j = 0; j < input_size2; j++) {
				loss_input[cur_featureMap][i][j] = 0;
			}
		}
	}

	for (int cur_featureMap = 0; cur_featureMap < num_of_inputs; cur_featureMap++) { //per input
		for (int i = 0; i < input_size1 - window; i += stride) {
			//per region
			for (int j = 0; j < input_size2 - window; j += stride) {
				// per region

				//matrix max pooling
				float max = (*input)[cur_featureMap][i][j];
				int indexX = 0;
				int indexY = 0;
				for (int m = 0; m < window; m++) {
					for (int n = 0; n < window; n++) {
						if (max < (*input)[cur_featureMap][i + m][j + n]) {
							max = (*input)[cur_featureMap][i + m][j + n];
							indexX = m;
							indexY = n;
						}
					}
				}

				//set only the lossInput of the "pixel" max pool kept
				loss_input[cur_featureMap][i + indexX][j + indexY] = (*loss_gradient)[cur_featureMap][i / stride][j / stride];
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
//void MaxPool::forwardJob(int packet) {
//	for (int i = packet * packetSize; i < (packet + 1) * packetSize; i++) {
//		int index1 = i / (output_size1 * output_size2);
//		int index2 = (i / output_size2) % output_size1 * stride;
//		int index3 = i % output_size2 * stride;
//
//		//matrix max pooling
//		float max = (*curr_input)[index1][index2][index3];
//		for (int m = 0; m < window; m++) {
//			for (int n = 0; n < window; n++)
//				if (max < (*curr_input)[index1][index2 + m][index3 + n])
//					max = (*curr_input)[index1][index2 + m][index3 + n];
//
//			(*curr_output)[index1][index2 / stride][index3 / stride] = max;
//		}
//	}
//	sem.V(1);
//}
//
// 
///**
// * Description
// *
// * @param packet
// * @return
//*/
//void MaxPool::forwardJobCleanup(int packet) {
//	for (int i = packet * packetSize; i < num_of_inputs * output_size1 * output_size2; i++) {
//		int index1 = i / (output_size1 * output_size2);
//		int index2 = (i / output_size2) % output_size1 * stride;
//		int index3 = i % output_size2 * stride;
//
//		//matrix max pooling
//		float max = (*curr_input)[index1][index2][index3];
//		for (int m = 0; m < window; m++) {
//			for (int n = 0; n < window; n++)
//				if (max < (*curr_input)[index1][index2 + m][index3 + n])
//					max = (*curr_input)[index1][index2 + m][index3 + n];
//
//			(*curr_output)[index1][index2 / stride][index3 / stride] = max;
//		}
//	}
//}
//
// /**
// * Description
// *
// * @param input
// * @return
//*/
//vector<vector<vector<float>>> MaxPool::forward(vector<vector<vector<float>>> &input) {
//	vector<vector<vector<float>>> output(num_of_inputs, vector<vector<float>>(output_size1, vector<float>(output_size2, 0.0)));
//	curr_output = &output;
//	curr_input = &input;
//
//	sem.set(0);
//	pool.setMaxPool(*this);
//				pool.setTask(3);
//	for (int i = 0; i < packets; i++) {
//		pushJob(i);
//	}
//	if ((num_of_inputs * output_size1 * output_size2) % packets != 0) {
//		forwardJobCleanup(packets + 1);
//	}
//	sem.P(packets);
//
//	return output;
//}
//
// /**
// * Description
// *
// * @param packet
// * @return
//*/
//void MaxPool::backpropJob(int packet) {
//	for (int i = packet * packetSize; i < (packet + 1) * packetSize; i++) {
//		int index1 = i / (output_size1 * output_size2);
//		int index2 = (i / output_size2) % output_size1 * stride;
//		int index3 = i % output_size2 * stride;
//
//		//matrix max pooling
//		float max = (*curr_input)[index1][index2][index3];
//		int indexX = 0;
//		int indexY = 0;
//		for (int m = 0; m < window; m++) {
//			for (int n = 0; n < window; n++) {
//				if (max < (*curr_input)[index1][index2 + m][index3 + n]) {
//					max = (*curr_input)[index1][index2 + m][index3 + n];
//					indexX = m;
//					indexY = n;
//				}
//			}
//		}
//
//		//set only the lossInput of the "pixel" max pool kept
//		(*curr_loss_input)[index1][index2 + indexX][index3 + indexY] = (*curr_loss_gradient)[index1][index2 / stride][index3 / stride];
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
//void MaxPool::backpropJobCleanup(int packet) {
//	for (int i = packet * packetSize; i < num_of_inputs * output_size1 * output_size2; i++) {
//		int index1 = i / (output_size1 * output_size2);
//		int index2 = (i / output_size2) % output_size1 * stride;
//		int index3 = i % output_size2 * stride;
//
//		//matrix max pooling
//		float max = (*curr_input)[index1][index2][index3];
//		int indexX = 0;
//		int indexY = 0;
//		for (int m = 0; m < window; m++) {
//			for (int n = 0; n < window; n++) {
//				if (max < (*curr_input)[index1][index2 + m][index3 + n]) {
//					max = (*curr_input)[index1][index2 + m][index3 + n];
//					indexX = m;
//					indexY = n;
//				}
//			}
//		}
//
//		//set only the lossInput of the "pixel" max pool kept
//		(*curr_loss_input)[index1][index2 + indexX][index3 + indexY] = (*curr_loss_gradient)[index1][index2 / stride][index3 / stride];
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
//vector<vector<vector<float>>> MaxPool::backprop(vector<vector<vector<float>>> &loss_gradient, vector<vector<vector<float>>> &last_input) {
//	vector<vector<vector<float>>> loss_input(num_of_inputs, vector<vector<float>>(input_size1, vector<float>(input_size2, 0.0)));
//	curr_loss_input = &loss_input;
//	curr_loss_gradient = &loss_gradient;
//	curr_input = &last_input;
//
//	sem.set(0);
//	pool.setMaxPool(*this);
//			pool.setTask(4);
//	for (int i = 0; i < packets; i++) {
//		pushJob(i);
//	}
//	if ((num_of_inputs * output_size1 * output_size2) % packets != 0) {
//		backpropJobCleanup(packets + 1);
//	}
//	sem.P(packets);
//
//	return loss_input;
//}
