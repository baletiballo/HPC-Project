#include "Conv.h"

Conv::Conv(){
	filters = new float [num_filters] [conv_size1] [conv_size2];
	biases = new float [num_filters];
	output = new float [num_filters] [num_windowsX] [num_windowsY];
	filter_gradient = new float [num_filters] [conv_size1] [conv_size2];
	bias_gradient = new float [num_filters];
	loss_input = new float [imageSizeX] [imageSizeY];

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

/**
 * Forward
 *
 * @param inputP
 * @return
 */
void Conv::forward(float inputP [imageSizeX] [imageSizeY]) {
	input = inputP;
	for (int cur_filter = 0; cur_filter < num_filters; cur_filter++) { //per filter		
		for (int i = 0; i < num_windowsX; i++) {//per region
			for (int j = 0; j < num_windowsY; j++) {// per region
				output[cur_filter][i][j] = biases[cur_filter];

				//set output at i j for the input representation cur_featureMap when filter cur_filter is applied
				//matrix multiplication and summation
				//DEPENDS ON: conv_size1(m), conv_size2(n)
				//output[cur_filter][i][j] += input[i + m][j + n] * filters[cur_filter][m][n];
				output[cur_filter][i][j] += input[i + 0][j + 0] * filters[cur_filter][0][0]; 
				output[cur_filter][i][j] += input[i + 0][j + 1] * filters[cur_filter][0][1];
				output[cur_filter][i][j] += input[i + 0][j + 2] * filters[cur_filter][0][2];

				output[cur_filter][i][j] += input[i + 1][j + 0] * filters[cur_filter][1][0];
				output[cur_filter][i][j] += input[i + 1][j + 1] * filters[cur_filter][1][1];
				output[cur_filter][i][j] += input[i + 1][j + 2] * filters[cur_filter][1][2];

				output[cur_filter][i][j] += input[i + 2][j + 0] * filters[cur_filter][2][0];
				output[cur_filter][i][j] += input[i + 2][j + 1] * filters[cur_filter][2][1];
				output[cur_filter][i][j] += input[i + 2][j + 2] * filters[cur_filter][2][2];
			}
		}
	}
}

/**
 * Call this after every Batch, addition of the gradients throughout a Batch is now done directly in backprop
 *
 */
void Conv::cleanup() {
	for (int cur_filter = 0; cur_filter < num_filters; cur_filter++) {
		bias_gradient[cur_filter] = 0;
		for (int i = 0; i < conv_size1; i++) {
			for (int j = 0; j < conv_size2; j++) {
				filter_gradient[cur_filter][i][j] = 0.0;
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
void Conv::backprop(float loss_gradientP [num_filters] [num_windowsX] [num_windowsY]) {
	loss_gradient = loss_gradientP;

	//zero the loss Input, since the same method to just add them all together cannot be applied here
	for (int i = 0; i < imageSizeX; i++) {
		for (int j = 0; j < imageSizeY; j++) {
			loss_input[i][j] = 0.0;
		}
	}

	for (int cur_filter = 0; cur_filter < num_filters; cur_filter++) { //per filter
		for (int i = 0; i < num_windowsX; i++) {//per region
			for (int j = 0; j < num_windowsY; j++) {// per region
				//matrix multiplication and summation
				//DEPENDS ON: conv_size1(m), conv_size2(n)
				//loss_input[i + m][j + n] += loss_gradient[cur_filter][i][j] * filters[cur_filter][m][n];
				//filter_gradient[cur_filter][m][n] += loss_gradient[cur_filter][i][j] * input[i + m][j + n];

				filter_gradient[cur_filter][0][0] 	+= loss_gradient[cur_filter][i][j] * input[i + 0][j + 0];
				filter_gradient[cur_filter][0][1] 	+= loss_gradient[cur_filter][i][j] * input[i + 0][j + 1];
				filter_gradient[cur_filter][0][2] 	+= loss_gradient[cur_filter][i][j] * input[i + 0][j + 2];

				filter_gradient[cur_filter][1][0] 	+= loss_gradient[cur_filter][i][j] * input[i + 1][j + 0];
				filter_gradient[cur_filter][1][1] 	+= loss_gradient[cur_filter][i][j] * input[i + 1][j + 1];
				filter_gradient[cur_filter][1][2] 	+= loss_gradient[cur_filter][i][j] * input[i + 1][j + 2];

				filter_gradient[cur_filter][2][0] 	+= loss_gradient[cur_filter][i][j] * input[i + 2][j + 0];
				filter_gradient[cur_filter][2][1] 	+= loss_gradient[cur_filter][i][j] * input[i + 2][j + 1];
				filter_gradient[cur_filter][2][2] 	+= loss_gradient[cur_filter][i][j] * input[i + 2][j + 2];


				loss_input[i + 0][j + 0] 			+= loss_gradient[cur_filter][i][j] * filters[cur_filter][0][0];				
				loss_input[i + 0][j + 1] 			+= loss_gradient[cur_filter][i][j] * filters[cur_filter][0][1];
				loss_input[i + 0][j + 2] 			+= loss_gradient[cur_filter][i][j] * filters[cur_filter][0][2];	

				loss_input[i + 1][j + 0] 			+= loss_gradient[cur_filter][i][j] * filters[cur_filter][1][0];				
				loss_input[i + 1][j + 1] 			+= loss_gradient[cur_filter][i][j] * filters[cur_filter][1][1];				
				loss_input[i + 1][j + 2] 			+= loss_gradient[cur_filter][i][j] * filters[cur_filter][1][2];	
							
				loss_input[i + 2][j + 0] 			+= loss_gradient[cur_filter][i][j] * filters[cur_filter][2][0];				
				loss_input[i + 2][j + 1] 			+= loss_gradient[cur_filter][i][j] * filters[cur_filter][2][1];				
				loss_input[i + 2][j + 2] 			+= loss_gradient[cur_filter][i][j] * filters[cur_filter][2][2];

				bias_gradient[cur_filter] += loss_gradient[cur_filter][i][j];
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
void Conv::forwardJob(int packet) {
	for (int index = packet * packetSizeForw; index < (packet + 1) * packetSizeForw; index++) {
		int cur_filter = (index / (num_windowsX * num_windowsY)) % num_filters;
		int i = (index / num_windowsY) % num_windowsX;
		int j = index % num_windowsY;

		output[cur_filter][i][j] = biases[cur_filter];

		//set output at i j for the input representation cur_featureMap when filter cur_filter is applied
		//matrix multiplication and summation
		//DEPENDS ON: conv_size1(m), conv_size2(n)
		//output[cur_filter][i][j] += input[i + m][j + n] * filters[cur_filter][m][n];
		output[cur_filter][i][j] += input[i + 0][j + 0] * filters[cur_filter][0][0];
		output[cur_filter][i][j] += input[i + 0][j + 1] * filters[cur_filter][0][1];
		output[cur_filter][i][j] += input[i + 0][j + 2] * filters[cur_filter][0][2];

		output[cur_filter][i][j] += input[i + 1][j + 0] * filters[cur_filter][1][0];
		output[cur_filter][i][j] += input[i + 1][j + 1] * filters[cur_filter][1][1];
		output[cur_filter][i][j] += input[i + 1][j + 2] * filters[cur_filter][1][2];

		output[cur_filter][i][j] += input[i + 2][j + 0] * filters[cur_filter][2][0];
		output[cur_filter][i][j] += input[i + 2][j + 1] * filters[cur_filter][2][1];
		output[cur_filter][i][j] += input[i + 2][j + 2] * filters[cur_filter][2][2];
	}

	sem.V(1);
}

/**
 * Description
 *
 * @param packet
 * @return
 */
void Conv::forwardJobCleanup(int packet) {
	for (int index = packet * packetSizeForw; index < num_filters * num_windowsX * num_windowsY; index++) {
		int cur_filter = (index / (num_windowsX * num_windowsY)) % num_filters;
		int i = (index / num_windowsY) % num_windowsX;
		int j = index % num_windowsY;

		output[cur_filter][i][j] = biases[cur_filter];

		//set output at i j for the input representation cur_featureMap when filter cur_filter is applied
		//matrix multiplication and summation
		//DEPENDS ON: conv_size1(m), conv_size2(n)
		//output[cur_filter][i][j] += input[i + m][j + n] * filters[cur_filter][m][n];
		output[cur_filter][i][j] += input[i + 0][j + 0] * filters[cur_filter][0][0];
		output[cur_filter][i][j] += input[i + 0][j + 1] * filters[cur_filter][0][1];
		output[cur_filter][i][j] += input[i + 0][j + 2] * filters[cur_filter][0][2];

		output[cur_filter][i][j] += input[i + 1][j + 0] * filters[cur_filter][1][0];
		output[cur_filter][i][j] += input[i + 1][j + 1] * filters[cur_filter][1][1];
		output[cur_filter][i][j] += input[i + 1][j + 2] * filters[cur_filter][1][2];

		output[cur_filter][i][j] += input[i + 2][j + 0] * filters[cur_filter][2][0];
		output[cur_filter][i][j] += input[i + 2][j + 1] * filters[cur_filter][2][1];
		output[cur_filter][i][j] += input[i + 2][j + 2] * filters[cur_filter][2][2];
	}
}

/**
 * Description
 *
 * @param inputP
 * @return
 */
void Conv::forward_par(float inputP [imageSizeX] [imageSizeY]) {
	input = inputP;

	sem.set(0);
	pool.setConv(*this);
	pool.setTask(1);
	for (int i = 0; i < num_packets; i++)
		pushJob(i);
	
	if (needForwCleanup)
		forwardJobCleanup(num_packets + 1);
	
	sem.P(num_packets);
}

/**
 * Description
 *
 * @param packet
 * @return
 */
void Conv::backpropJob(int packet) {
	for (int index = packet * packetSizeBack; index < (packet + 1) * packetSizeBack; index++) {
		int i = (index / num_windowsY) % num_windowsX;
		int j = index % num_windowsY;
		loss_input[i][j] = 0.0;

		//DEPENDS ON: num_filters
		//loss_input[i][j] += loss_gradient[cur_filter][i - m][j - n] * filters[cur_filter][m][n];
		for (int m = 0; m < std::min(i + 1, conv_size1); m++)
			for (int n = 0; n < std::min(j + 1, conv_size2); n++)
				loss_input[i][j] += loss_gradient[0][i - m][j - n] * filters[0][m][n];

		for (int m = 0; m < std::min(i + 1, conv_size1); m++)
			for (int n = 0; n < std::min(j + 1, conv_size2); n++)
				loss_input[i][j] += loss_gradient[1][i - m][j - n] * filters[1][m][n];

		for (int m = 0; m < std::min(i + 1, conv_size1); m++)
			for (int n = 0; n < std::min(j + 1, conv_size2); n++)
				loss_input[i][j] += loss_gradient[2][i - m][j - n] * filters[2][m][n];

		for (int m = 0; m < std::min(i + 1, conv_size1); m++)
			for (int n = 0; n < std::min(j + 1, conv_size2); n++)
				loss_input[i][j] += loss_gradient[3][i - m][j - n] * filters[3][m][n];

		for (int m = 0; m < std::min(i + 1, conv_size1); m++)
			for (int n = 0; n < std::min(j + 1, conv_size2); n++)
				loss_input[i][j] += loss_gradient[4][i - m][j - n] * filters[4][m][n];

		for (int m = 0; m < std::min(i + 1, conv_size1); m++)
			for (int n = 0; n < std::min(j + 1, conv_size2); n++)
				loss_input[i][j] += loss_gradient[5][i - m][j - n] * filters[5][m][n];

		for (int m = 0; m < std::min(i + 1, conv_size1); m++)
			for (int n = 0; n < std::min(j + 1, conv_size2); n++)
				loss_input[i][j] += loss_gradient[6][i - m][j - n] * filters[6][m][n];

		for (int m = 0; m < std::min(i + 1, conv_size1); m++)
			for (int n = 0; n < std::min(j + 1, conv_size2); n++)
				loss_input[i][j] += loss_gradient[7][i - m][j - n] * filters[7][m][n];
	}

	for (int cur_filter = packet; cur_filter < num_filters; cur_filter += num_packets) {
		for (int i = 0; i < num_windowsX; i++) {//per region		
			for (int j = 0; j < num_windowsY; j++) {// per region				
				//matrix multiplication and summation
				//DEPENDS ON: conv_size1(m), conv_size2(n)
				//filter_gradient[cur_filter][m][n] += loss_gradient[cur_filter][i][j] * input[i + m][j + n];
				filter_gradient[cur_filter][0][0] += loss_gradient[cur_filter][i][j] * input[i + 0][j + 0];
				filter_gradient[cur_filter][0][1] += loss_gradient[cur_filter][i][j] * input[i + 0][j + 1];
				filter_gradient[cur_filter][0][2] += loss_gradient[cur_filter][i][j] * input[i + 0][j + 2];

				filter_gradient[cur_filter][1][0] += loss_gradient[cur_filter][i][j] * input[i + 1][j + 0];
				filter_gradient[cur_filter][1][1] += loss_gradient[cur_filter][i][j] * input[i + 1][j + 1];
				filter_gradient[cur_filter][1][2] += loss_gradient[cur_filter][i][j] * input[i + 1][j + 2];

				filter_gradient[cur_filter][2][0] += loss_gradient[cur_filter][i][j] * input[i + 2][j + 0];
				filter_gradient[cur_filter][2][1] += loss_gradient[cur_filter][i][j] * input[i + 2][j + 1];
				filter_gradient[cur_filter][2][2] += loss_gradient[cur_filter][i][j] * input[i + 2][j + 2];

				bias_gradient[cur_filter] += loss_gradient[cur_filter][i][j];
			}
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
void Conv::backpropJobCleanup(int packet) {
	for (int index = packet * packetSizeBack; index < num_windowsX * num_windowsY; index++) {
		int i = (index / num_windowsY) % num_windowsX;
		int j = index % num_windowsY;
		loss_input[i][j] = 0.0;

		//DEPENDS ON: num_filters
		//loss_input[i][j] += loss_gradient[cur_filter][i - m][j - n] * filters[cur_filter][m][n];
		for (int m = 0; m < std::min(i + 1, conv_size1); m++) 
			for (int n = 0; n < std::min(j + 1, conv_size2); n++) 
				loss_input[i][j] += loss_gradient[0][i - m][j - n] * filters[0][m][n];

		for (int m = 0; m < std::min(i + 1, conv_size1); m++)
			for (int n = 0; n < std::min(j + 1, conv_size2); n++)
				loss_input[i][j] += loss_gradient[1][i - m][j - n] * filters[1][m][n];

		for (int m = 0; m < std::min(i + 1, conv_size1); m++)
			for (int n = 0; n < std::min(j + 1, conv_size2); n++)
				loss_input[i][j] += loss_gradient[2][i - m][j - n] * filters[2][m][n];

		for (int m = 0; m < std::min(i + 1, conv_size1); m++)
			for (int n = 0; n < std::min(j + 1, conv_size2); n++)
				loss_input[i][j] += loss_gradient[3][i - m][j - n] * filters[3][m][n];

		for (int m = 0; m < std::min(i + 1, conv_size1); m++)
			for (int n = 0; n < std::min(j + 1, conv_size2); n++)
				loss_input[i][j] += loss_gradient[4][i - m][j - n] * filters[4][m][n];

		for (int m = 0; m < std::min(i + 1, conv_size1); m++)
			for (int n = 0; n < std::min(j + 1, conv_size2); n++)
				loss_input[i][j] += loss_gradient[5][i - m][j - n] * filters[5][m][n];

		for (int m = 0; m < std::min(i + 1, conv_size1); m++)
			for (int n = 0; n < std::min(j + 1, conv_size2); n++)
				loss_input[i][j] += loss_gradient[6][i - m][j - n] * filters[6][m][n];

		for (int m = 0; m < std::min(i + 1, conv_size1); m++)
			for (int n = 0; n < std::min(j + 1, conv_size2); n++)
				loss_input[i][j] += loss_gradient[7][i - m][j - n] * filters[7][m][n];
	}

	for (int cur_filter = packet; cur_filter < num_filters; cur_filter += num_packets) {
		for (int i = 0; i < num_windowsX; i++) { //per region
			for (int j = 0; j < num_windowsY; j++) { // per region
				//matrix multiplication and summation
				//DEPENDS ON: conv_size1(m), conv_size2(n)
				//filter_gradient[cur_filter][m][n] += loss_gradient[cur_filter][i][j] * input[i + m][j + n];
				filter_gradient[cur_filter][0][0] += loss_gradient[cur_filter][i][j] * input[i + 0][j + 0];
				filter_gradient[cur_filter][0][1] += loss_gradient[cur_filter][i][j] * input[i + 0][j + 1];
				filter_gradient[cur_filter][0][2] += loss_gradient[cur_filter][i][j] * input[i + 0][j + 2];

				filter_gradient[cur_filter][1][0] += loss_gradient[cur_filter][i][j] * input[i + 1][j + 0];
				filter_gradient[cur_filter][1][1] += loss_gradient[cur_filter][i][j] * input[i + 1][j + 1];
				filter_gradient[cur_filter][1][2] += loss_gradient[cur_filter][i][j] * input[i + 1][j + 2];

				filter_gradient[cur_filter][2][0] += loss_gradient[cur_filter][i][j] * input[i + 2][j + 0];
				filter_gradient[cur_filter][2][1] += loss_gradient[cur_filter][i][j] * input[i + 2][j + 1];
				filter_gradient[cur_filter][2][2] += loss_gradient[cur_filter][i][j] * input[i + 2][j + 2];

				bias_gradient[cur_filter] += loss_gradient[cur_filter][i][j];
			}
		}
	}
}

/**
 * Description
 *
 * @param loss_gradientP
 * @return
 */
void Conv::backprop_par(float loss_gradientP [num_filters] [num_windowsX] [num_windowsY]) {
	loss_gradient = loss_gradientP;

	sem.set(0);
	pool.setConv(*this);
	pool.setTask(2);
	for (int i = 0; i < num_packets; i++)
		pushJob(i);
	
	if (needBackCleanup)
		backpropJobCleanup(num_packets + 1);
	
	sem.P(num_packets);
}