/*
 * cnn.cpp
 *
 *  Created on: 03.05.2021
 *      Author: Stefan, Hannah, Silas
 */

#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <random>
#include <tuple>
#include <chrono>
#include <ctime> 

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>

using namespace std;

class JobQueue {
public:
	mutex jobQueueMutex;
	queue<packaged_task<void()>> jobQueue;
	condition_variable cv;
	bool abort = false;

	void push(packaged_task<void()> job) {
		unique_lock<mutex> l(jobQueueMutex);
		jobQueue.push(move(job));
		cv.notify_one();
	}

	packaged_task<void()> pop() {
		unique_lock<mutex> l(jobQueueMutex);
		cv.wait(l, [this] {
			return abort || !jobQueue.empty();
		});
		if (abort)
			return {};
		auto r = move(jobQueue.front());
		jobQueue.pop();
		return move(r);
	}

	void terminate() {
		unique_lock<mutex> l(jobQueueMutex);
		abort = true;
		queue<packaged_task<void()>> emptyQueue;
		emptyQueue.swap(jobQueue);
		cv.notify_all();
	}
};

class ThreadPool {
public:
	int threads;
	vector<thread> pool;
	JobQueue queue;

	ThreadPool(int numThreads) {
		threads = numThreads;

		for (int i = 0; i < threads; i++) {
			pool.push_back(thread(&threadsDoWork, this));
		}
	}

	void threadsDoWork() {
		while (true) {
			packaged_task<void()> job = queue.pop();
			if (queue.abort == true) {
				break;
			}
			job();
		}
	}
};

ThreadPool pool(8);

void endThreads() {
	pool.queue.terminate();
}

void pushJob(packaged_task<void()> job) {
	pool.queue.push(move(job));
}

class Sem {
public:
	mutex mtx;
	condition_variable cv;
	int count;

	Sem(int countInit) {
		count = countInit;
	}

	void V(int n) {
		unique_lock<mutex> lck(mtx);
		count += n;
		cv.notify_all();
	}

	void P(int n) {
		unique_lock<mutex> lck(mtx);
		while (count < n) {
			cv.wait(lck);
		}

		count = -n;
	}

	void set(int n) {
		unique_lock<mutex> lck(mtx);
		count = n;
		cv.notify_all();
	}
};

Sem sem(0);

vector<float> flatten(vector<vector<vector<float>>> &t1) {
	vector<float> out(t1.size() * t1[0].size() * t1[0][0].size());
	for (unsigned i = 0; i < t1.size(); i++) {
		for (unsigned j = 0; j < t1[0].size(); j++) {
			for (unsigned k = 0; k < t1[0][0].size(); k++) {
				out[i * t1[0].size() * t1[0][0].size() + j * t1[0][0].size() + k] = t1[i][j][k];
			}
		}
	}
	return out;
}

vector<vector<vector<float>>> deflatten(vector<float> &t1, int s1, int s2, int s3) {
	vector<vector<vector<float>>> out(s1, vector<vector<float>>(s2, vector<float>(s3)));
	for (int i = 0; i < s1; i++) {
		for (int j = 0; j < s2; j++) {
			for (int k = 0; k < s3; k++) {
				out[i][j][k] = t1[i * s2 * s3 + j * s3 + k];
			}
		}
	}
	return out;
}

void softmax(vector<float> &t1, vector<float> &output) { //t1 and output have to have the same size
	float sum = 0.0;
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = exp(t1[i]);
		sum += output[i];
	}
	for (unsigned i = 0; i < t1.size(); i++) {
		output[i] = output[i] / sum;
	}
}

int reluPackets = 64;
int reluPacketSize;
int reluSize1;
int reluSize2;
int reluSize3;
vector<vector<vector<float>>> *relu_input = nullptr;
vector<vector<vector<float>>> *relu_input_2 = nullptr;

void ReLuJob(int packet) {
	for (int i = packet * reluPacketSize; i < (packet + 1) * reluPacketSize; i++) {
		int index1 = i / (reluSize2 * reluSize3);
		int index2 = (i / reluSize3) % reluSize2;
		int index3 = i % reluSize3;
		if ((*relu_input)[index1][index2][index3] <= 0) {
			(*relu_input)[index1][index2][index3] = 0;
		}
	}
	sem.V(1);
}

void ReLuJobCleanup(int packet) {
	for (int i = packet * reluPacketSize; i < (reluSize1 * reluSize2 * reluSize3); i++) {
		int index1 = i / (reluSize2 * reluSize3);
		int index2 = (i / reluSize3) % reluSize2;
		int index3 = i % reluSize3;
		if ((*relu_input)[index1][index2][index3] <= 0) {
			(*relu_input)[index1][index2][index3] = 0;
		}
	}
	sem.V(1);
}

void ReLu(vector<vector<vector<float>>> &t1) {
	reluSize1 = t1.size();
	reluSize2 = t1[0].size();
	reluSize3 = t1[0][0].size();
	reluPacketSize = (reluSize1 * reluSize2 * reluSize3) / reluPackets;
	relu_input = &t1;

	sem.set(0);
	for (int i = 0; i < reluPackets; i++) {
		packaged_task<void()> job(bind(&ReLuJob, i));
		pushJob(move(job));
	}
	if ((reluSize1 * reluSize2 * reluSize3) % reluPackets != 0) {
		ReLuJobCleanup(reluPackets + 1);
	}
	sem.P(reluPackets);
}

void ReLuPrimeJob(int packet) {
	for (int i = packet * reluPacketSize; i < (packet + 1) * reluPacketSize; i++) {
		int index1 = i / (reluSize2 * reluSize3);
		int index2 = (i / reluSize3) % reluSize2;
		int index3 = i % reluSize3;
		if ((*relu_input_2)[index1][index2][index3] <= 0) {
			(*relu_input)[index1][index2][index3] = 0;
		}
	}
	sem.V(1);
}

void ReLuJobPrimeCleanup(int packet) {
	for (int i = packet * reluPacketSize; i < (reluSize1 * reluSize2 * reluSize3); i++) {
		int index1 = i / (reluSize2 * reluSize3);
		int index2 = (i / reluSize3) % reluSize2;
		int index3 = i % reluSize3;
		if ((*relu_input_2)[index1][index2][index3] <= 0) {
			(*relu_input)[index1][index2][index3] = 0;
		}
	}
	sem.V(1);
}

void ReLuPrime(vector<vector<vector<float>>> &t1, vector<vector<vector<float>>> &t2) {
	reluSize1 = t1.size();
	reluSize2 = t1[0].size();
	reluSize3 = t1[0][0].size();
	reluPacketSize = (reluSize1 * reluSize2 * reluSize3) / reluPackets;
	relu_input = &t1;
	relu_input_2 = &t2;

	sem.set(0);
	for (int i = 0; i < reluPackets; i++) {
		packaged_task<void()> job(bind(&ReLuPrimeJob, i));
		pushJob(move(job));
	}
	if ((reluSize1 * reluSize2 * reluSize3) % reluPackets != 0) {
		ReLuJobPrimeCleanup(reluPackets + 1);
	}
	sem.P(reluPackets);
}

/*void ReLu(vector<vector<vector<float>>> &t1) {
	for (unsigned i = 0; i < t1.size(); i++) {
		for (unsigned j = 0; j < t1[i].size(); j++) {
			for (unsigned k = 0; k < t1[i][j].size(); k++) {
				if (t1[i][j][k] <= 0) {
					t1[i][j][k] = 0;
				}
			}
		}
	}
}

void ReLuPrime(vector<vector<vector<float>>> &t1, vector<vector<vector<float>>> &t2) {
	for (unsigned i = 0; i < t1.size(); i++) {
		for (unsigned j = 0; j < t1[i].size(); j++) {
			for (unsigned k = 0; k < t1[i][j].size(); k++) {
				if (t2[i][j][k] <= 0) {
					t1[i][j][k] = 0;
				}
			}
		}
	}
}*/

class Conv {
public:
	int num_filters;
	int num_of_inputs, input_size1, input_size2;
	int conv_size1, conv_size2;
	int num_windows1, num_windows2;
	vector<vector<vector<float>>> filters;
	vector<float> biases;
	vector<vector<vector<float>>> *curr_input = nullptr;
	vector<vector<vector<float>>> *curr_output = nullptr;
	vector<vector<vector<float>>> *curr_loss_gradient = nullptr;
	vector<vector<vector<float>>> *curr_filter_gradient = nullptr;
	vector<float> *curr_bias_gradient = nullptr;
	vector<vector<vector<float>>> *curr_loss_input = nullptr;
	int packets = 64;
	int packetSize;

	Conv(int f, int c1, int c2, int n, int s1, int s2) {
		num_filters = f;
		conv_size1 = c1;
		conv_size2 = c2;
		num_of_inputs = n;
		input_size1 = s1;
		input_size2 = s2;
		num_windows1 = input_size1 - conv_size1 + 1;
		num_windows2 = input_size2 - conv_size2 + 1;
		packetSize = (num_filters * num_of_inputs * num_windows1 * num_windows2) / packets;

		filters.resize(num_filters, vector<vector<float>>(conv_size1, vector<float>(conv_size2)));
		biases.resize(num_filters, 0.0);

		normal_distribution<float> distribution(0.0, 1.0);
		random_device dev;
		default_random_engine generator(dev());
		for (int cur_filter = 0; cur_filter < num_filters; cur_filter++) {
			for (int i = 0; i < conv_size1; i++) {
				for (int j = 0; j < conv_size2; j++) {
					filters[cur_filter][i][j] = distribution(generator) / (conv_size1 * conv_size2);
				}
			}
		}
	}

	void forwardJob(int packet) {
		for (int i = packet * packetSize; i < (packet + 1) * packetSize; i++) {
			int index1 = i / (num_windows1 * num_windows2);
			int index2 = (i / num_windows2) % num_windows1;
			int index3 = i % num_windows2;
			(*curr_output)[index1][index2][index3] = biases[index1 / num_of_inputs];

			//set output at i j for the input representation cur_featureMap when filter cur_filter is applied
			//matrix multiplication and summation
			for (int m = 0; m < conv_size1; m++) {
				for (int n = 0; n < conv_size2; n++) {
					(*curr_output)[index1][index2][index3] += (*curr_input)[index1 % num_of_inputs][index2 + m][index3 + n]
							* filters[index1 / num_of_inputs][m][n];
				}
			}
		}
		sem.V(1);
	}

	void forwardJobCleanup(int packet) {
		for (int i = packet * packetSize; i < num_filters * num_of_inputs * num_windows1 * num_windows2; i++) {
			int index1 = i / (num_windows1 * num_windows2);
			int index2 = (i / num_windows2) % num_windows1;
			int index3 = i % num_windows2;
			int filter = index1 / num_of_inputs;
			int featureMap = index1 % num_of_inputs;

			(*curr_output)[index1][index2][index3] = biases[filter];

			//set output at i j for the input representation cur_featureMap when filter cur_filter is applied
			//matrix multiplication and summation
			for (int m = 0; m < conv_size1; m++) {
				for (int n = 0; n < conv_size2; n++) {
					(*curr_output)[index1][index2][index3] += (*curr_input)[featureMap][index2 + m][index3 + n] * filters[filter][m][n];
				}
			}
		}
	}

	vector<vector<vector<float>>> forward(vector<vector<vector<float>>> &input) {
		vector<vector<vector<float>>> output(num_of_inputs * num_filters, vector<vector<float>>(num_windows1, vector<float>(num_windows2)));
		curr_output = &output;
		curr_input = &input;

		sem.set(0);
		for (int i = 0; i < packets; i++) {
			packaged_task<void()> job(bind(&forwardJob, this, i));
			pushJob(move(job));
		}
		if ((num_filters * num_of_inputs * num_windows1 * num_windows2) % packets != 0) {
			forwardJobCleanup(packets + 1);
		}
		sem.P(packets);

		return output;
	}

	void backpropJob(int packet) {
		for (int i = packet * packetSize; i < (packet + 1) * packetSize; i++) {
			int index1 = i / (num_windows1 * num_windows2);
			int index2 = (i / num_windows2) % num_windows1;
			int index3 = i % num_windows2;
			int filter = index1 / num_of_inputs;
			int featureMap = index1 % num_of_inputs;

			// per region
			//matrix multiplication and summation
			for (int m = 0; m < conv_size1; m++) {
				for (int n = 0; n < conv_size2; n++) {
					(*curr_filter_gradient)[filter][m][n] += (*curr_loss_gradient)[index1][index2][index3] * (*curr_input)[featureMap][index2 + m][index3 + n];
					(*curr_loss_input)[featureMap][index2 + m][index3 + n] += (*curr_loss_gradient)[index1][index2][index3] * filters[filter][m][n];
				}
			}

			(*curr_bias_gradient)[filter] += (*curr_loss_gradient)[index1][index2][index3];
		}
		sem.V(1);
	}

	void backpropJobCleanup(int packet) {
		for (int i = packet * packetSize; i < num_filters * num_of_inputs * num_windows1 * num_windows2; i++) {
			int index1 = i / (num_windows1 * num_windows2);
			int index2 = (i / num_windows2) % num_windows1;
			int index3 = i % num_windows2;
			int filter = index1 / num_of_inputs;
			int featureMap = index1 % num_of_inputs;

			// per region
			//matrix multiplication and summation
			for (int m = 0; m < conv_size1; m++) {
				for (int n = 0; n < conv_size2; n++) {
					(*curr_filter_gradient)[filter][m][n] += (*curr_loss_gradient)[index1][index2][index3] * (*curr_input)[featureMap][index2 + m][index3 + n];
					(*curr_loss_input)[featureMap][index2 + m][index3 + n] += (*curr_loss_gradient)[index1][index2][index3] * filters[filter][m][n];
				}
			}

			(*curr_bias_gradient)[filter] += (*curr_loss_gradient)[index1][index2][index3];
		}
	}

	tuple<vector<vector<vector<float>>>, vector<float>, vector<vector<vector<float>>>> backprop(vector<vector<vector<float>>> &loss_gradient,
			vector<vector<vector<float>>> &last_input) {
		vector<vector<vector<float>>> filter_gradient(num_filters, vector<vector<float>>(conv_size1, vector<float>(conv_size2, 0.0)));
		vector<float> bias_gradient(num_filters, 0.0);
		vector<vector<vector<float>>> loss_input(num_of_inputs, vector<vector<float>>(input_size1, vector<float>(input_size2, 0.0)));

		curr_filter_gradient = &filter_gradient;
		curr_bias_gradient = &bias_gradient;
		curr_loss_input = &loss_input;
		curr_loss_gradient = &loss_gradient;
		curr_input = &last_input;

		sem.set(0);
		for (int i = 0; i < packets; i++) {
			packaged_task<void()> job(bind(&backpropJob, this, i));
			pushJob(move(job));
		}
		if ((num_filters * num_of_inputs * num_windows1 * num_windows2) % packets != 0) {
			backpropJobCleanup(packets + 1);
		}
		sem.P(packets);

		return {filter_gradient, bias_gradient, loss_input};
	}
};

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

class FullyConnectedLayer {
public:
	unsigned num_of_inputs, input_size1, input_size2;
	unsigned num_weights;
	unsigned total_size;
	vector<vector<float>> weights;
	vector<float> biases;
	vector<float> *curr_input = nullptr;
	vector<float> *curr_output = nullptr;
	vector<float> *curr_loss_gradient = nullptr;
	vector<vector<float>> *curr_weight_gradient = nullptr;
	vector<float> *curr_bias_gradient = nullptr;
	vector<float> *curr_loss_input = nullptr;
	int packets = 64;
	int packetSize;
	deque<mutex> mtx;
	deque<mutex> mtx_big;

	FullyConnectedLayer(unsigned w, unsigned n, unsigned s1, unsigned s2) {
		num_weights = w;
		num_of_inputs = n;
		input_size1 = s1;
		input_size2 = s2;
		total_size = num_of_inputs * input_size1 * input_size2;
		weights.resize(num_weights, vector<float>(total_size));
		biases.resize(num_weights, 0.0);
		packetSize = (num_weights * total_size) / packets;
		mtx.resize(num_weights);
		mtx_big.resize(total_size);

		normal_distribution<float> distribution(0.0, 1.0);
		for (unsigned i = 0; i < num_weights; i++) {

			for (unsigned j = 0; j < total_size; j++) {
				random_device dev;
				default_random_engine generator(dev());
				weights[i][j] = distribution(generator) / (total_size);
			}
		}
	}

	void forwardJob(int packet) {
		float tmp = 0.0;
		for (int i = packet * packetSize; i < (packet + 1) * packetSize; i++) {
			int index1 = i / (total_size);
			int index2 = i % total_size;

			tmp += (*curr_input)[index2] * weights[index1][index2];

			if ((unsigned) index2 + 1 == total_size) {
				mtx[index1].lock();
				(*curr_output)[index1] += tmp;
				mtx[index1].unlock();

				tmp = 0.0;
			}
		}
		if (tmp != 0.0) {
			int index1 = (((packet + 1) * packetSize) - 1) / (total_size);
			mtx[index1].lock();
			(*curr_output)[index1] += tmp;
			mtx[index1].unlock();
		}
		sem.V(1);
	}

	void forwardJobCleanup(int packet) {
		float tmp = 0.0;
		for (int i = packet * packetSize; (unsigned) i < num_weights * total_size; i++) {
			int index1 = i / (total_size);
			int index2 = i % total_size;

			tmp += (*curr_input)[index2] * weights[index1][index2];

			if ((unsigned) index2 + 1 == total_size) {
				mtx[index1].lock();
				(*curr_output)[index1] += tmp;
				mtx[index1].unlock();

				tmp = 0.0;
			}
		}
		if (tmp != 0.0) {
			int index1 = (((packet + 1) * packetSize) - 1) / (total_size);
			mtx[index1].lock();
			(*curr_output)[index1] += tmp;
			mtx[index1].unlock();
		}
	}

	vector<float> forward(vector<float> &input) {
		vector<float> output(num_weights);
		curr_input = &input;
		curr_output = &output;

		sem.set(0);
		for (int i = 0; i < packets; i++) {
			packaged_task<void()> job(bind(&forwardJob, this, i));
			pushJob(move(job));
		}
		if ((num_weights * total_size) % packets != 0) {
			forwardJobCleanup(packets + 1);
		}
		sem.P(packets);

		for (unsigned i = 0; i < num_weights; i++) {
			output[i] += biases[i];
		}
		return output;
	}

	void backpropJob(int packet) {
		for (int i = packet * packetSize; i < (packet + 1) * packetSize; i++) {
			int index1 = i / (total_size);
			int index2 = i % total_size;

			(*curr_weight_gradient)[index1][index2] = (*curr_loss_gradient)[index1] * (*curr_input)[index2];

			mtx_big[index2].lock();
			(*curr_loss_input)[index2] += weights[index1][index2] * (*curr_loss_gradient)[index1];
			mtx_big[index2].unlock();
		}
		sem.V(1);
	}

	void backpropJobCleanup(int packet) {
		for (int i = packet * packetSize; (unsigned) i < num_weights * total_size; i++) {
			int index1 = i / (total_size);
			int index2 = i % total_size;

			(*curr_weight_gradient)[index1][index2] = (*curr_loss_gradient)[index1] * (*curr_input)[index2];

			mtx_big[index2].lock();
			(*curr_loss_input)[index2] += weights[index1][index2] * (*curr_loss_gradient)[index1];
			mtx_big[index2].unlock();
		}
	}

	tuple<vector<vector<float>>, vector<float>, vector<float>> backprop(vector<float> &loss_gradient, vector<float> &last_input) {
		vector<vector<float>> weight_gradient(num_weights, vector<float>(total_size));
		vector<float> bias_gradient(num_weights);
		vector<float> loss_input(total_size, 0.0);

		curr_weight_gradient = &weight_gradient;
		curr_bias_gradient = &bias_gradient;
		curr_loss_input = &loss_input;
		curr_loss_gradient = &loss_gradient;
		curr_input = &last_input;

		sem.set(0);
		for (int i = 0; i < packets; i++) {
			packaged_task<void()> job(bind(&backpropJob, this, i));
			pushJob(move(job));
		}
		if ((num_weights * total_size) % packets != 0) {
			backpropJobCleanup(packets + 1);
		}
		for (unsigned i = 0; i < num_weights; i++) { //fast enough as is
			(*curr_bias_gradient)[i] = (*curr_loss_gradient)[i];
		}
		sem.P(packets);

		return {weight_gradient, bias_gradient, loss_input};
	}
};

class CNN {
public:
	static const int sizeX = 28;
	static const int sizeY = 28;
	static const int num_conv_layers = 1;
	const int num_filters = 8;
	const int pool_layers_window = 2;
	const int pool_layers_stride = 2;
	const int conv_size1 = 3;
	const int conv_size2 = 3;
	const int num_weights = 10;
	const float EPSILON = 1.0f * pow(10.0f, -8);

	vector<Conv> conv_layers;
	vector<MaxPool> pooling_layers;
	FullyConnectedLayer *connected_layer;

	vector<vector<vector<vector<float>>>> first_momentum_filters;
	vector<vector<vector<vector<float>>>> second_momentum_filters;
	vector<vector<float>> first_momentum_conv_biases;
	vector<vector<float>> second_momentum_conv_biases;
	vector<vector<float>> first_momentum_weights;
	vector<vector<float>> second_momentum_weights;
	vector<float> first_momentum_conn_biases;
	vector<float> second_momentum_conn_biases;

	CNN() {
		int currX = sizeX;
		int currY = sizeY;
		int images = 1;
		for (unsigned i = 0; i < num_conv_layers; i++) {
			vector<vector<vector<float>>> firstMomentum(num_filters, vector<vector<float>>(conv_size1, vector<float>(conv_size2, 0.0)));
			first_momentum_filters.push_back(firstMomentum);
			vector<vector<vector<float>>> secondMomentum(num_filters, vector<vector<float>>(conv_size1, vector<float>(conv_size2, 0.0)));
			second_momentum_filters.push_back(secondMomentum);
			vector<float> firstMomentumBiases(num_filters, 0.0);
			first_momentum_conv_biases.push_back(firstMomentumBiases);
			vector<float> secondMomentumBiases(num_filters, 0.0);
			second_momentum_conv_biases.push_back(secondMomentumBiases);

			conv_layers.push_back(Conv(num_filters, conv_size1, conv_size2, images, currX, currY));
			currX -= (conv_size1 - 1);
			currY -= (conv_size2 - 1);
			images *= num_filters;
			pooling_layers.push_back(MaxPool(pool_layers_window, pool_layers_stride, images, currX, currY));
			currX = (currX - pool_layers_window) / pool_layers_stride + 1;
			currY = (currY - pool_layers_window) / pool_layers_stride + 1;
		}

		connected_layer = new FullyConnectedLayer(num_weights, images, currX, currY);
		first_momentum_weights.resize((num_weights), vector<float>(images * currX * currY, 0.0));
		second_momentum_weights.resize((num_weights), vector<float>(images * currX * currY, 0.0));
		first_momentum_conn_biases.resize(num_weights, 0.0);
		second_momentum_conn_biases.resize(num_weights, 0.0);
	}

	tuple<float, int, tuple<vector<vector<vector<vector<float>>>>, vector<vector<float>>, vector<vector<float>>, vector<float>>> conv(
			vector<vector<vector<float>>> &image, int label) {
		vector<vector<vector<vector<float>>>> z;
		z.push_back(image);
		for (int i = 0; i < num_conv_layers; i++) {
			vector<vector<vector<float>>> help = conv_layers[i].forward(z.back());
			ReLu(help);
			z.push_back(help);
			help = pooling_layers[i].forward(z.back());
			z.push_back(help);
		}
		vector<float> h = flatten(z.back());
		vector<float> res = (*connected_layer).forward(h);

		softmax(res, res);

		float loss = -log(res[label]);
		int correct = 0;

		int argmax = 0;
		for (int i = 0; i < num_weights; i++) {
			if (res[i] >= res[argmax]) {
				argmax = i;
			}
		}

		if (argmax == label)
			correct = 1;

		res[label] -= 1;

		vector<vector<vector<vector<float>>>> filter_gradients;
		vector<vector<float>> conv_bias_gradients;
		vector<vector<float>> weight_gradient;
		vector<float> conn_bias_gradient;

		tuple<vector<vector<float>>, vector<float>, vector<float>> helpconn = (*connected_layer).backprop(res, h);
		weight_gradient = get<0>(helpconn);
		conn_bias_gradient = get<1>(helpconn);
		vector<vector<vector<float>>> helpback = deflatten(get<2>(helpconn), (*connected_layer).num_of_inputs, (*connected_layer).input_size1,
				(*connected_layer).input_size2);
		for (int i = num_conv_layers - 1; i > -1; i--) {
			helpback = pooling_layers[i].backprop(helpback, z[2 * i + 1]);
			ReLuPrime(helpback, z[2 * i + 1]);
			tuple<vector<vector<vector<float>>>, vector<float>, vector<vector<vector<float>>>> helpconv = conv_layers[i].backprop(helpback, z[2 * i]);
			filter_gradients.push_back(get<0>(helpconv));
			conv_bias_gradients.push_back(get<1>(helpconv));
			helpback = get<2>(helpconv);

		}

		return {loss, correct, {filter_gradients, conv_bias_gradients, weight_gradient, conn_bias_gradient}};
	}

	tuple<float, float> learn(float alpha, float beta1, float beta2, vector<vector<vector<float>> > &x_batch, vector<int> &y_batch, int batchSize, int step) {
		vector<vector<vector<float>> > image(1, vector<vector<float>>(sizeX, vector<float>(sizeY)));
		image[0] = x_batch[0];
		int label = y_batch[0];
		tuple<float, int, tuple<vector<vector<vector<vector<float>>>>, vector<vector<float>>, vector<vector<float>>, vector<float>>> convRes = conv(image,
				label);
		tuple<vector<vector<vector<vector<float>>>>, vector<vector<float>>, vector<vector<float>>, vector<float>> convResGra = get<2>(convRes);
		vector<vector<vector<vector<float>>>> filterGradients = get<0>(convResGra);
		vector<vector<float>> filterBiases = get<1>(convResGra);
		vector<vector<float>> weightGradient = get<2>(convResGra);
		vector<float> weightBiases = get<3>(convResGra);

		float loss = get<0>(convRes);
		float correct = get<1>(convRes);

		for (int i = 1; i < batchSize; i++) {
			image[0] = x_batch[i];
			label = y_batch[i];

			convRes = conv(image, label);
			loss += get<0>(convRes);
			correct += get<1>(convRes);

			convResGra = get<2>(convRes);
			addGradients(get<0>(convResGra), get<1>(convResGra), get<2>(convResGra), get<3>(convResGra), filterGradients, filterBiases, weightGradient,
					weightBiases);
		}

//normalize
		/*multiply(filterGradients, 1.0 / batchSize, filterGradients);
		 multiply(filterBiases, 1.0 / batchSize, filterBiases);
		 multiply(weightGradient, 1.0 / batchSize, weightGradient);
		 multiply(weightBiases, 1.0 / batchSize, weightBiases);*///now done in updateFilters and updateWeights
//ADAM learning
		updateFilters(filterGradients, filterBiases, beta1, beta2, alpha, -1, batchSize);
		updateWeights(weightGradient, weightBiases, beta1, beta2, alpha, -1, batchSize);

//SGD learning
		/*updateFilters2(alpha, filterGradients, filterBiases, batchSize);
		 updateWeights2(alpha, weightGradient, weightBiases, batchSize);*/

		return {loss, correct};
	}

	void addGradients(vector<vector<vector<vector<float>>>> &t1, vector<vector<float>> &t2, vector<vector<float>> &t3, vector<float> &t4,
			vector<vector<vector<vector<float>>>> &o1, vector<vector<float>> &o2, vector<vector<float>> &o3, vector<float> &o4) {
		for (unsigned i = 0; i < t1.size(); i++) {
			for (unsigned j = 0; j < t1[i].size(); j++) {
				for (unsigned k = 0; k < t1[i][j].size(); k++) {
					for (unsigned l = 0; l < t1[i][j][k].size(); l++) {
						o1[i][j][k][l] += t1[i][j][k][l];
					}
				}
				o2[i][j] += t2[i][j];
			}
		}

		for (unsigned i = 0; i < t3.size(); i++) {
			for (unsigned j = 0; j < t3[i].size(); j++) {
				o3[i][j] += t3[i][j];
			}
			o4[i] += t4[i];
		}
	}

	void updateFilters(vector<vector<vector<vector<float>>>> &filterGradients, vector<vector<float>> &filterBiases, float beta1, float beta2, float alpha,
			int step, int batchSize) {
		float corr1 = 1;
		float corr2 = 1;
		if (step != -1) { //if we dont want to enable this correction
			corr1 = 1 - pow(beta1, step);
			corr2 = 1 - pow(beta2, step);
		}

		for (unsigned i = 0; i < first_momentum_filters.size(); i++) {
			for (unsigned j = 0; j < first_momentum_filters[i].size(); j++) {
				for (unsigned k = 0; k < first_momentum_filters[i][j].size(); k++) {
					for (unsigned l = 0; l < first_momentum_filters[i][j][k].size(); l++) {
						first_momentum_filters[i][j][k][l] = beta1 * first_momentum_filters[i][j][k][l]
								+ (1.0 - beta1) * filterGradients[i][j][k][l] / batchSize;
						second_momentum_filters[i][j][k][l] = beta1 * second_momentum_filters[i][j][k][l]
								+ (1.0 - beta1) * pow(filterGradients[i][j][k][l] / batchSize, 2);

						conv_layers[i].filters[j][k][l] = conv_layers[i].filters[j][k][l]
								- alpha * ((first_momentum_filters[i][j][k][l] / corr1) / (sqrt(second_momentum_filters[i][j][k][l] / corr2) + EPSILON));
					}
				}
				first_momentum_conv_biases[i][j] = beta1 * first_momentum_conv_biases[i][j] + (1.0 - beta1) * filterBiases[i][j] / batchSize;
				second_momentum_conv_biases[i][j] = beta1 * second_momentum_conv_biases[i][j] + (1.0 - beta1) * pow(filterBiases[i][j] / batchSize, 2);

				conv_layers[i].biases[j] = conv_layers[i].biases[j]
						- alpha * ((first_momentum_conv_biases[i][j] / corr1) / (sqrt(second_momentum_conv_biases[i][j] / corr2) + EPSILON));
			}
		}
	}

	void updateWeights(vector<vector<float>> &weightGradient, vector<float> &weightBiases, float beta1, float beta2, float alpha, int step, int batchSize) {
		float corr1 = 1;
		float corr2 = 1;
		if (step != -1) { //if we dont want to enable this correction
			corr1 = 1 - pow(beta1, step);
			corr2 = 1 - pow(beta2, step);
		}

		for (unsigned i = 0; i < first_momentum_weights.size(); i++) {
			for (unsigned j = 0; j < first_momentum_weights[i].size(); j++) {
				first_momentum_weights[i][j] = beta1 * first_momentum_weights[i][j] + (1.0 - beta1) * weightGradient[i][j] / batchSize;
				second_momentum_weights[i][j] = beta1 * second_momentum_weights[i][j] + (1.0 - beta1) * pow(weightGradient[i][j] / batchSize, 2);

				(*connected_layer).weights[i][j] = (*connected_layer).weights[i][j]
						- alpha * ((first_momentum_weights[i][j] / corr1) / (sqrt(second_momentum_weights[i][j] / corr2) + EPSILON));
			}
			first_momentum_conn_biases[i] = beta1 * first_momentum_conn_biases[i] + (1.0 - beta1) * weightBiases[i] / batchSize;
			second_momentum_conn_biases[i] = beta1 * second_momentum_conn_biases[i] + (1.0 - beta1) * pow(weightBiases[i] / batchSize, 2);

			(*connected_layer).biases[i] = (*connected_layer).biases[i]
					- alpha * ((first_momentum_conn_biases[i] / corr1) / (sqrt(second_momentum_conn_biases[i] / corr2) + EPSILON));
		}
	}

	/*void updateWeights2(float alpha, vector<vector<float>> &weightGradient, vector<float> &weightBiases, int batchSize) {
		for (unsigned i = 0; i < weightGradient.size(); i++) {
			for (unsigned j = 0; j < weightGradient[i].size(); j++) {
				(*connected_layer).weights[i][j] -= alpha * weightGradient[i][j] / batchSize;
			}
			(*connected_layer).biases[i] -= alpha * weightBiases[i] / batchSize;
		}
	}

	void updateFilters2(float alpha, vector<vector<vector<vector<float>>>> &filterGradients, vector<vector<float>> &filterBiases, int batchSize) {
		for (unsigned i = 0; i < first_momentum_filters.size(); i++) {
			for (unsigned j = 0; j < first_momentum_filters[i].size(); j++) {
				for (unsigned k = 0; k < first_momentum_filters[i][j].size(); k++) {
					for (unsigned l = 0; l < first_momentum_filters[i][j][k].size(); l++) {
						conv_layers[i].filters[j][k][l] -= alpha * filterGradients[i][j][k][l] / batchSize;
					}
				}
				conv_layers[i].biases[j] -= alpha * filterBiases[i][j] / batchSize;
			}
		}
	}*/

};

void read_trainingData(string filename, vector<vector<float>> &training_images, vector<int> &correct_lables);

int main() {
	try {
		vector<vector<float>> training_images(42000, vector<float>(784));
		vector<int> correct_lables(42000);

		/* Einlesen der Trainingsdaten*/
		read_trainingData("train.txt", training_images, correct_lables);

		/*Vorbereiten des Netzwerks für das Training*/
		const int batchSize = 32;
		const int imageSize = 28;
		const int num_steps = 3000;

		float endLoss;
		float endCorr;

		vector<vector<vector<float>>> batch_images(batchSize, vector<vector<float>>(imageSize, vector<float>(imageSize)));
		vector<int> batch_lables(batchSize);
		CNN cnn; //Das benutzte Netzwerk. Topologieänderungen bitte in der Klasse CNN
		const float alpha = 0.001; //Lernrate
		const float beta1 = 0.9; //Erstes Moment
		const float beta2 = 0.999; //Zweites Moment
		cout << "Beginn des Trainings\n";
		auto training_startTime = chrono::system_clock::now(); // Interner Timer um die Laufzeit zu messen

		for (int i = 0; i < num_steps; i++) {

			/* Vorbereiten des Trainingsbatches */
			int randIndex = rand() % (42000 - batchSize);
			for (unsigned j = 0; j < batchSize; j++) { //erstelle einen zufälligen Batch für das Training
				for (int k = 0; k < 784; k++) //Reformatierung des flachen Vektors in Zeilen und Spalten
					batch_images[j][k / imageSize][k % imageSize] = training_images[j + randIndex][k];

				batch_lables[j] = correct_lables[j + randIndex];
			}

			tuple<float, float> res = cnn.learn(alpha, beta1, beta2, batch_images, batch_lables, batchSize, i + 1);

			float loss = get<0>(res);
			float correct = get<1>(res) * 1.0;

			if(i % 500 == 0){//Zwischenupdates. Nur alle paar hundert Baches, um Konsole übersichtlich zu halten
				cout << "Batch " << i + 1 << " \t Average Loss " << loss / batchSize << "\t Accuracy " << correct / batchSize << "\n";
			}

			if (num_steps - i <= 10) {
				endLoss += loss;
				endCorr += correct;
			}
		}

		auto training_endTime = chrono::system_clock::now();
		chrono::duration<double> totalTime = training_endTime - training_startTime;
		cout << "Total time: " << (int) (totalTime.count() / 60) << " minutes " << (int) (totalTime.count()) % 60 << " seconds\n";
		cout << "Average loss in last " << batchSize * 10 << " tries:" << endLoss / (10 * batchSize) << "\t Average accuracy in last 10 batches: "
				<< endCorr / (10 * batchSize) << "\n";
		endThreads();

		return 0;
	} catch (const exception&) {
		endThreads();
		return -1;
	}
}

void read_trainingData(string filename, vector<vector<float>> &training_images, vector<int> &correct_lables) {
	ifstream myFile(filename);
	if (myFile.is_open()) {
		cout << "Lese Trainingsdaten ein";
		int lineNum = 0;
		string line;
		while (getline(myFile, line)) {
			istringstream ss(line);
			string token;
			int i = 0;
			while (getline(ss, token, '\t')) {
				int digit = stoi(token, nullptr);
				if (i == 0)			//erste Zahl jeder Zeile ist das lable
					correct_lables[lineNum] = digit;
				else
					//der Rest das Graustufenbild
					training_images[lineNum][i - 1] = static_cast<float>(digit) / static_cast<float>(255);

				i++;
			}

			lineNum++;
		}
		myFile.close();
	}

}
