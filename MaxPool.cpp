
#include <vector>
#include "ParallelStuff.cpp"

using namespace std;

class MaxPool {
public:
	int num_of_inputs, input_size1, input_size2;
	int window, stride;
	int output_size1, output_size2;
	vector<vector<vector<float>>> *curr_input = nullptr;
	vector<vector<vector<float>>> *curr_output = nullptr;
	vector<vector<vector<float>>> *curr_loss_gradient = nullptr;
	vector<vector<vector<float>>> *curr_loss_input = nullptr;
	int packets = 64;
	int packetSize;

	MaxPool(int w, int s, int n, int s1, int s2) {
		window = w;
		stride = s;
		num_of_inputs = n;
		input_size1 = s1;
		input_size2 = s2;
		output_size1 = (input_size1 - window) / stride + 1;
		output_size2 = (input_size2 - window) / stride + 1;
		packetSize = (num_of_inputs * output_size1 * output_size2) / packets;
	}

	void forwardJob(int packet) {
		for (int i = packet * packetSize; i < (packet + 1) * packetSize; i++) {
			int index1 = i / (output_size1 * output_size2);
			int index2 = (i / output_size2) % output_size1 * stride;
			int index3 = i % output_size2 * stride;

			//matrix max pooling
			float max = (*curr_input)[index1][index2][index3];
			for (int m = 0; m < window; m++) {
				for (int n = 0; n < window; n++)
					if (max < (*curr_input)[index1][index2 + m][index3 + n])
						max = (*curr_input)[index1][index2 + m][index3 + n];

				(*curr_output)[index1][index2 / stride][index3 / stride] = max;
			}
		}
		sem.V(1);
	}

	void forwardJobCleanup(int packet) {
		for (int i = packet * packetSize; i < num_of_inputs * output_size1 * output_size2; i++) {
			int index1 = i / (output_size1 * output_size2);
			int index2 = (i / output_size2) % output_size1 * stride;
			int index3 = i % output_size2 * stride;

			//matrix max pooling
			float max = (*curr_input)[index1][index2][index3];
			for (int m = 0; m < window; m++) {
				for (int n = 0; n < window; n++)
					if (max < (*curr_input)[index1][index2 + m][index3 + n])
						max = (*curr_input)[index1][index2 + m][index3 + n];

				(*curr_output)[index1][index2 / stride][index3 / stride] = max;
			}
		}
	}

	vector<vector<vector<float>>> forward(vector<vector<vector<float>>> &input) {
		vector<vector<vector<float>>> output(num_of_inputs, vector<vector<float>>(output_size1, vector<float>(output_size2, 0.0)));
		curr_output = &output;
		curr_input = &input;

		sem.set(0);
		for (int i = 0; i < packets; i++) {
			packaged_task<void()> job(bind(&forwardJob, this, i));
			pushJob(move(job));
		}
		if ((num_of_inputs * output_size1 * output_size2) % packets != 0) {
			forwardJobCleanup(packets + 1);
		}
		sem.P(packets);

		return output;
	}

	void backpropJob(int packet) {
		for (int i = packet * packetSize; i < (packet + 1) * packetSize; i++) {
			int index1 = i / (output_size1 * output_size2);
			int index2 = (i / output_size2) % output_size1 * stride;
			int index3 = i % output_size2 * stride;

			//matrix max pooling
			float max = (*curr_input)[index1][index2][index3];
			int indexX = 0;
			int indexY = 0;
			for (int m = 0; m < window; m++) {
				for (int n = 0; n < window; n++) {
					if (max < (*curr_input)[index1][index2 + m][index3 + n]) {
						max = (*curr_input)[index1][index2 + m][index3 + n];
						indexX = m;
						indexY = n;
					}
				}
			}

			//set only the lossInput of the "pixel" max pool kept
			(*curr_loss_input)[index1][index2 + indexX][index3 + indexY] = (*curr_loss_gradient)[index1][index2 / stride][index3 / stride];
		}
		sem.V(1);
	}

	void backpropJobCleanup(int packet) {
		for (int i = packet * packetSize; i < num_of_inputs * output_size1 * output_size2; i++) {
			int index1 = i / (output_size1 * output_size2);
			int index2 = (i / output_size2) % output_size1 * stride;
			int index3 = i % output_size2 * stride;

			//matrix max pooling
			float max = (*curr_input)[index1][index2][index3];
			int indexX = 0;
			int indexY = 0;
			for (int m = 0; m < window; m++) {
				for (int n = 0; n < window; n++) {
					if (max < (*curr_input)[index1][index2 + m][index3 + n]) {
						max = (*curr_input)[index1][index2 + m][index3 + n];
						indexX = m;
						indexY = n;
					}
				}
			}

			//set only the lossInput of the "pixel" max pool kept
			(*curr_loss_input)[index1][index2 + indexX][index3 + indexY] = (*curr_loss_gradient)[index1][index2 / stride][index3 / stride];
		}
	}

	vector<vector<vector<float>>> backprop(vector<vector<vector<float>>> &loss_gradient, vector<vector<vector<float>>> &last_input) {
		vector<vector<vector<float>>> loss_input(num_of_inputs, vector<vector<float>>(input_size1, vector<float>(input_size2, 0.0)));
		curr_loss_input = &loss_input;
		curr_loss_gradient = &loss_gradient;
		curr_input = &last_input;

		sem.set(0);
		for (int i = 0; i < packets; i++) {
			packaged_task<void()> job(bind(&backpropJob, this, i));
			pushJob(move(job));
		}
		if ((num_of_inputs * output_size1 * output_size2) % packets != 0) {
			backpropJobCleanup(packets + 1);
		}
		sem.P(packets);

		return loss_input;
	}
};
