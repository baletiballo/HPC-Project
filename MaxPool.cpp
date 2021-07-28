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
	needCleanup=(num_of_inputs * output_size1 * output_size2) % packets != 0;

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
		for (int m = 0; m < window; m++) {
			for (int n = 0; n < window; n++)
				if (max < (*input)[cur_featureMap][indexX + m][indexY + n])
					max = (*input)[cur_featureMap][indexX + m][indexY + n];

			output[cur_featureMap][indexX / stride][indexY / stride] = max;
		}
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

		//matrix max pooling
		float max = (*input)[cur_featureMap][indexX][indexY];
		for (int m = 0; m < window; m++) {
			for (int n = 0; n < window; n++)
				if (max < (*input)[cur_featureMap][indexX + m][indexY + n])
					max = (*input)[cur_featureMap][indexX + m][indexY + n];

			output[cur_featureMap][indexX / stride][indexY / stride] = max;
		}
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
		int i = (index / output_size2) % output_size1;
		int j = index % output_size2;
		loss_input[cur_featureMap][i][j] = 0;
	}

	for (int index = packet * packetSize; index < (packet + 1) * packetSize; index++) {
		int cur_featureMap = index / (output_size1 * output_size2);
		int i = (index / output_size2) % output_size1 * stride;
		int j = index % output_size2 * stride;

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
		int i = (index / output_size2) % output_size1;
		int j = index % output_size2;
		loss_input[cur_featureMap][i][j] = 0;
	}

	for (int index = packet * packetSize; index < num_of_inputs * output_size1 * output_size2; index++) {
		int cur_featureMap = index / (output_size1 * output_size2);
		int i = (index / output_size2) % output_size1 * stride;
		int j = index % output_size2 * stride;

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
