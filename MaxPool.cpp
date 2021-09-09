#include "MaxPool.h"

using namespace std;

MaxPool::MaxPool() {
	output = new float [threads][num_filters] [output_size1] [output_size2]{};
	loss_input = new float [threads][num_inputs] [input_size1] [input_size2]{};
}

void MaxPool::setLossGradient(float loss_gradientP [threads][num_filters] [output_size1] [output_size2]) {
	loss_gradient = loss_gradientP;
}

void MaxPool::setInput(float inputP [threads][num_filters] [input_size1] [input_size2]) {
	input = inputP;
}

void MaxPool::forward(int_fast8_t spot) {

	for (int cur_featureMap = 0; cur_featureMap < num_inputs; cur_featureMap++) { //per input
		for (int i = 0; i < input_size1 - window; i += stride) {
			//per region
			for (int j = 0; j < input_size2 - window; j += stride) {
				// per region

				//matrix max pooling
				float max = input[spot][cur_featureMap][i][j];
				int maxX = 0;
				int maxY = 0;

				for (int m = 0; m < pool_layers_window; m++){
					for (int n = 0; n < pool_layers_window; n++){
						if (max < input[spot][cur_featureMap][i + m][j + n]) {
							max = input[spot][cur_featureMap][i + m][j + n];
							maxX = m;
							maxY = n;
						 }
					}
				}
				
				output[spot][cur_featureMap][i / stride][j / stride] = max;

				//Koordinaten schreiben
				get<0>(inputCoordsOfOutputPixels[spot][cur_featureMap][i / stride + j / stride * output_size1]) = maxX;
				get<1>(inputCoordsOfOutputPixels[spot][cur_featureMap][i / stride + j / stride * output_size1]) = maxY;
			}
		}
	}
}

void MaxPool::backprop(int_fast8_t spot) {

	for (int cur_featureMap = 0; cur_featureMap < num_inputs; cur_featureMap++) { //per input
		for (int i = 0; i < input_size1 - window; i += stride) {
			//per region
			for (int j = 0; j < input_size2 - window; j += stride) {
				// per region

				//zero the loss Input
				const int previousIndexX = get<0>(previouslyUsedLossInputPixels[spot][cur_featureMap][i / stride + j / stride * output_size1]);
				const int previousIndexY = get<1>(previouslyUsedLossInputPixels[spot][cur_featureMap][i / stride + j / stride * output_size1]);
				loss_input[spot][cur_featureMap][i + previousIndexX][j + previousIndexY] = 0.0;

				//matrix max pooling
				const int indexX = get<0>(inputCoordsOfOutputPixels[spot][cur_featureMap][i / stride + j / stride * output_size1]);
				const int indexY = get<1>(inputCoordsOfOutputPixels[spot][cur_featureMap][i / stride + j / stride * output_size1]);

				//set only the lossInput of the "pixel" max pool kept
				loss_input[spot][cur_featureMap][i + indexX][j + indexY] = loss_gradient[spot][cur_featureMap][i / stride][j / stride];

				get<0>(previouslyUsedLossInputPixels[spot][cur_featureMap][i / stride + j / stride * output_size1]) = indexX;
				get<1>(previouslyUsedLossInputPixels[spot][cur_featureMap][i / stride + j / stride * output_size1]) = indexY;
			}
		}
	}
}