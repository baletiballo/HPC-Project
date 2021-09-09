#include "FullyConnectedLayer.h"

FullyConnectedLayer::FullyConnectedLayer() {
	//mtx.resize(num_classes);

	weights = new float[num_classes][num_lastLayer_inputNeurons] { };
	biases = new float[num_classes] { };

	output = new float[threads][num_classes] { };
	weight_gradient = new float[threads][num_classes][num_lastLayer_inputNeurons] { };
	bias_gradient = new float[threads][num_classes] { };
	loss_input = new float[threads][num_inputs][input_size1][input_size2] { };

	std::normal_distribution<float> distribution(0.0, 1.0);
	for (unsigned i = 0; i < num_classes; i++) {
		for (unsigned j = 0; j < num_lastLayer_inputNeurons; j++) {
			std::random_device dev;
			std::default_random_engine generator(dev());
			weights[i][j] = distribution(generator) / (num_lastLayer_inputNeurons);
		}
	}
}

void FullyConnectedLayer::setLossGradient(float loss_gradientP[threads][num_classes]) {
	loss_gradient = loss_gradientP;
}

void FullyConnectedLayer::setInput(float inputP[threads][num_filters][input_size1][input_size2]) {
	input = inputP;
}

/**
 * Forward
 *
 * @param inputP
 */
void FullyConnectedLayer::forward(int_fast8_t spot) {
	for (unsigned currClass = 0; currClass < num_classes; currClass++) {
		output[spot][currClass] = biases[currClass];
		for (unsigned currFeatureMap = 0; currFeatureMap < num_inputs; currFeatureMap++) {
			for (unsigned k = 0; k < input_size1; k++) {
				for(unsigned l = 0; l < input_size2; l++){
					output[spot][currClass] += 
						input[spot][currFeatureMap][k][l] * weights[currClass][currFeatureMap * input_size1 * input_size2 + k * input_size2 + l];
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
	for (int spot = 0; spot < threads; spot++) {
		for (unsigned cur_weight = 0; cur_weight < num_classes; cur_weight++) {
			bias_gradient[spot][cur_weight] = 0.0;
			for (unsigned i = 0; i < num_lastLayer_inputNeurons; i++) {
				weight_gradient[spot][cur_weight][i] = 0.0;
			}
		}
	}
}

/**
 * Backprop (notice: gradients are getting already added here, since weight_gradient and bias_gradient only get reset to zero after a cleanup() call and are only needed after a batch)
 *
 * @param loss_gradientP
 */
void FullyConnectedLayer::backprop(int_fast8_t spot) {

	for (unsigned currFeatureMap = 0; currFeatureMap < num_inputs; currFeatureMap++) {
		for (unsigned currX = 0; currX < input_size1; currX++) {
			for (unsigned currY = 0; currY < input_size2; currY++) {
				//zero the loss Input, since the same method to just add them all together cannot be applied here
				loss_input[spot][currFeatureMap][currX][currY] = 0.0;

				for(int currClass = 0; currClass < num_classes; currClass++){
					loss_input[spot][currFeatureMap][currX][currY] += weights[currClass][currFeatureMap * input_size1 * input_size2 + currX * input_size2 + currY]
						* loss_gradient[spot][currClass];
					weight_gradient[spot][currClass][currFeatureMap * input_size1 * input_size2 + currX * input_size2 + currY] += loss_gradient[spot][currClass]
						* input[spot][currFeatureMap][currX][currY];
				}
			}
		}
	}

	for (unsigned i = 0; i < num_classes; i++) {
		bias_gradient[spot][i] += loss_gradient[spot][i];
	}

}