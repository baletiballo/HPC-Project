#include "Conv.h"

Conv::Conv() {
	filters = new float[num_filters][conv_size1][conv_size2] { };
	biases = new float[num_filters] { };

	output = new float[threads][num_filters][num_windowsX][num_windowsY] { };
	filter_gradient = new float[threads][num_filters][conv_size1][conv_size2] { };
	bias_gradient = new float[threads][num_filters] { };
	//loss_input = new float [imageSizeX] [imageSizeY]{};

	std::normal_distribution<float> distribution(0.0, 1.0);
	std::random_device dev;
	std::default_random_engine generator(dev());
	for (int cur_filter = 0; cur_filter < num_filters; cur_filter++) {
		for (int i = 0; i < conv_size1; i++) {
			for (int j = 0; j < conv_size2; j++) {
				filters[cur_filter][i][j] = distribution(generator) / (conv_size1 * conv_size2);
			}
		}
	}
}

void Conv::setLossGradient(float loss_gradientP[threads][num_filters][num_windowsX][num_windowsY]) {
	loss_gradient = loss_gradientP;
}

void Conv::setInput(float (*inputP) [imageSizeX][imageSizeY]) {
	input = inputP;
}

/**
 * Forward
 *
 * @param inputP
 * @return
 */
void Conv::forward(int_fast8_t spot, int image) {
	for (int cur_filter = 0; cur_filter < num_filters; cur_filter++) { //per filter
		for (int i = 0; i < num_windowsX; i++) { //per region
			for (int j = 0; j < num_windowsY; j++) { // per region
				output[spot][cur_filter][i][j] = biases[cur_filter];

				//set output at i j for the input representation cur_featureMap when filter cur_filter is applied
				//matrix multiplication and summation


				for(int m = 0; m < conv_size1; m++){
					for(int n = 0; n < conv_size2; n++){
						output[spot][cur_filter][i][j] += input[image][i + m][j + n] * filters[cur_filter][m][n];
					}
				}
				
			}
		}
	}
}

/**
 * Call this after every Batch, addition of the gradients throughout a Batch is now done directly in backprop
 *
 */
void Conv::cleanup() {
	for (int spot = 0; spot < threads; spot++) {
		for (int cur_filter = 0; cur_filter < num_filters; cur_filter++) {
			bias_gradient[spot][cur_filter] = 0.0;
			for (int i = 0; i < conv_size1; i++) {
				for (int j = 0; j < conv_size2; j++) {
					filter_gradient[spot][cur_filter][i][j] = 0.0;
				}
			}
		}
	}
}

/**
 * Backprop (notice: gradients are getting already added here, since filter_gradient and bias_gradient only get reset to zero after a cleanup() call and are only needed after a batch)
 *
 * @param loss_gradientP
 * @return
 */
void Conv::backprop(int_fast8_t spot, int image) {
	/*//zero the loss Input, since the same method to just add them all together cannot be applied here
	 for (int i = 0; i < imageSizeX; i++) {
	 for (int j = 0; j < imageSizeY; j++) {
	 loss_input[i][j] = 0.0;
	 }
	 }*/

	for (int cur_filter = 0; cur_filter < num_filters; cur_filter++) { //per filter
		for (int i = 0; i < num_windowsX; i++) { //per region
			for (int j = 0; j < num_windowsY; j++) { // per region
				//matrix multiplication and summation
				
				for(int m = 0; m < conv_size1; m++){
					for(int n = 0; n < conv_size2; n++){
						filter_gradient[spot][cur_filter][m][n] += loss_gradient[spot][cur_filter][i][j] * input[image][i + m][j + n];
					}
				}

				bias_gradient[spot][cur_filter] += loss_gradient[spot][cur_filter][i][j];
			}
		}
	}
}