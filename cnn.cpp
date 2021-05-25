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

using namespace std;

class Conv5x5
{
public:
	int num_filters;
	int size1, size2, size3;
	const int conv_size = 5;
	vector<vector<vector<float>>> filters;
	vector<float> biases;

	vector<vector<vector<float>>> last_input;

	Conv5x5(int n, int s1, int s2, int s3)
	{
		num_filters = n;
		size1 = s1;
		size2 = s2;
		size3 = s3;
		filters.resize(conv_size, vector<vector<float>>(conv_size, vector<float>(num_filters)));
		biases.resize(num_filters, 0.0);

		normal_distribution<float> distribution(0.0, 1.0);
		for (int i = 0; i < conv_size; i++)
		{
			for (int j = 0; j < conv_size; j++)
			{
				for (int cur_filter = 0; cur_filter < num_filters; cur_filter++)
				{
					random_device dev;
					default_random_engine generator(dev());
					filters[i][j][cur_filter] = distribution(generator) / 9;
				}
			}
		}
	}

	vector<vector<vector<float>>> forward(vector<vector<vector<float>>> input)
	{
		vector<vector<vector<float>>> output(size1 * num_filters, vector<vector<float>>(size2 - (conv_size-1), vector<float>(size3 - (conv_size-1))));
		for (int i = 0; i < size2 - 2; i++)
		{
			//per region
			for (int j = 0; j < size3 - 2; j++)
			{
				// per region
				for (int cur_filter = 0; cur_filter < num_filters; cur_filter++)
				{
					//per filter
					for (int cur_featureMap = 0; cur_featureMap < size1; cur_featureMap++)
					{
						//per passed representation
						output[cur_featureMap * num_filters + cur_filter][i][j] = biases[cur_filter];

						//set output at i j for the input representation cur_featureMap when filter cur_filter is applied
						//matrix multiplication and summation
						for (int m = 0; m < conv_size; m++)
							for (int n = 0; n < conv_size; n++)
								output[cur_featureMap * num_filters + cur_filter][i][j] += input[cur_featureMap][i + m][j + n] * filters[m][n][cur_filter];
					}
				}
			}
		}

		last_input = input;
		return output;
	}

	vector<vector<vector<float>>> backprop(vector<vector<vector<float>>> lossGradient, float learn_rate)
	{
		vector<vector<vector<float>>> filterGradient(3, vector<vector<float>>(3, vector<float>(num_filters, 0.0)));
		vector<float> filterBias(num_filters, 0.0);
		vector<vector<vector<float>>> lossInput(size1, vector<vector<float>>(size2, vector<float>(size3, 0.0)));

		for (int i = 0; i < size2 - 2; i++)
		{
			//per region
			for (int j = 0; j < size3 - 2; j++)
			{
				// per region
				for (int cur_filter = 0; cur_filter < num_filters; cur_filter++)
				{
					//per filter
					for (int cur_featureMap = 0; cur_featureMap < size1; cur_featureMap++)
					{
						//per passed representation
						//matrix multiplication and summation
						for (int m = 0; m < 3; m++)
						{
							for (int n = 0; n < 3; n++)
							{
								filterGradient[m][n][cur_filter] += lossGradient[cur_featureMap * num_filters + cur_filter][i][j] * last_input[cur_featureMap][i + m][j + n];
								lossInput[cur_featureMap][i + m][j + n] += lossGradient[cur_featureMap * num_filters + cur_filter][i][j] * filters[m][n][cur_filter];
							}
						}

						filterBias[cur_filter] += lossGradient[cur_featureMap * num_filters + cur_filter][i][j];
					}
				}
			}
		}

		for (int i = 0; i < num_filters; i++)
			biases[i] -= learn_rate * filterBias[i];

		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				for (int k = 0; k < num_filters; k++)
					filters[i][j][k] -= learn_rate * filterGradient[i][j][k];

		return lossInput;
	}
};

class MaxPool
{
public:
	int size1, size2, size3;
	int window, stride;

	vector<vector<vector<float>>> last_input;

	MaxPool(int w, int s, int s1, int s2, int s3)
	{
		window = w;
		stride = s;
		size1 = s1;
		size2 = s2;
		size3 = s3;
	}

	vector<vector<vector<float>>> forward(vector<vector<vector<float>>> input)
	{
		vector<vector<vector<float>>> output(size1, vector<vector<float>>((size2 - window) / stride + 1, vector<float>((size3 - window) / stride + 1)));
		for (int i = 0; i < size2 - window; i += stride)
		{
			//per region
			for (int j = 0; j < size3 - window; j += stride)
			{
				// per region
				for (int cur_featureMap = 0; cur_featureMap < size1; cur_featureMap++)
				{
					//per passed representation
					//matrix max pooling
					float max = input[cur_featureMap][i][j];
					for (int m = 0; m < window; m++)
					{
						for (int n = 0; n < window; n++)
							if (max < input[cur_featureMap][i + m][j + n])
								max = input[cur_featureMap][i + m][j + n];

						output[cur_featureMap][i / stride][j / stride] = max;
					}
				}
			}
		}

		last_input = input;
		return output;
	}

	vector<vector<vector<float>>> backprop(vector<vector<vector<float>>> lossGradient, float learn_rate)
	{
		vector<vector<vector<float>>> lossInput(size1, vector<vector<float>>(size2, vector<float>(size3, 0.0)));

		for (int i = 0; i < size2 - window; i += stride)
		{
			//per region
			for (int j = 0; j < size3 - window; j += stride)
			{
				// per region
				for (int cur_featureMap = 0; cur_featureMap < size1; cur_featureMap++)
				{
					//per passed representation
					//matrix max pooling
					float max = last_input[cur_featureMap][i][j];
					int indexX = 0;
					int indexY = 0;
					for (int m = 0; m < window; m++)
					{
						for (int n = 0; n < window; n++)
						{
							if (max < last_input[cur_featureMap][i + m][j + n])
							{
								max = last_input[cur_featureMap][i + m][j + n];
								indexX = m;
								indexY = n;
							}
						}
					}

					//set only the lossInput of the "pixel" max pool kept
					lossInput[cur_featureMap][i + indexX][j + indexY] = lossGradient[cur_featureMap][i][j];
				}
			}
		}

		return lossInput;
	}
};

class FullyConnectedLayer
{
public:
	int num_featureMaps;	//Number of feature maps the convolutional Layers generate
	int size2, size3;		//Dimensions of the feature maps
	static const int num_weights = 10; //Number of 
	vector<vector<float>> weights;
	vector<float> biases;

	vector<float> last_inputVector;
	vector<float> last_totals;
	float last_sum = 0.0;

	FullyConnectedLayer(int s1, int s2, int s3) {
		num_featureMaps = s1;
		size2 = s2;
		size3 = s3;
		weights.resize(num_featureMaps * size2 * size3, vector<float>(num_weights));
		biases.resize(num_weights, 0.0);

		normal_distribution<float> distribution(0.0, 1.0);
		for (int i = 0; i < num_featureMaps * size2 * size3; i++)
		{
			for (int j = 0; j < num_weights; j++)
			{
				random_device dev;
				default_random_engine generator(dev());
				weights[i][j] = distribution(generator) / 9;
			}
		}

		last_totals.resize(num_weights);
	}

	vector<float> forward(vector<vector<vector<float>>> input)
	{
		vector<float> output(num_weights);
		for (int i = 0; i < num_weights; i++)
			output[i] = biases[i];

		//flatten (the curve xD)
		vector<float> inputVector(num_featureMaps * size2 * size3);
		for (int i = 0; i < num_featureMaps; i++)
			for (int j = 0; j < size2; j++)
				for (int k = 0; k < size3; k++)
					inputVector[i * size2 * size3 + j * size3 + k] = input[i][j][k];

		for (int i = 0; i < num_featureMaps * size2 * size3; i++) //per feature
			for (int j = 0; j < num_weights; j++) //per weights
				output[j] += inputVector[i] * weights[i][j];

		last_inputVector = inputVector;

		//activation function
		float total = 0.0;
		for (int i = 0; i < num_weights; i++)
		{
			output[i] = exp(output[i]);
			last_totals[i] = output[i];
			total += output[i];
		}
		last_sum = total;

		//normalize
		for (int i = 0; i < num_weights; i++)
			output[i] = output[i] / total;

		return output;
	}

	vector<vector<vector<float>>> backprop(vector<float> lossGradient, float learn_rate)
	{
		vector<vector<vector<float>>> lossInput(num_featureMaps, vector<vector<float>>(size2, vector<float>(size3, 0.0)));

		int index = -1;
		for (int i = 0; i < num_weights; i++)
			if (lossGradient[i] < FLT_EPSILON) //TODO maybe change value if not a good fit
				index = i;
		
		const float gradient = lossGradient[index];

		float doutdt[num_weights];
		for (int i = 0; i < num_weights; i++)
			doutdt[i] = -last_totals[index] * last_totals[i] / (last_sum * last_sum);
			
		doutdt[index] = last_totals[index] * last_sum - last_totals[index] / (last_sum * last_sum);

		const auto dLdt = new float[num_weights];
		for (int i = 0; i < num_weights; i++)
		{
			dLdt[i] = gradient * doutdt[i];
			biases[i] -= learn_rate * dLdt[i];
		}
			
		for (int i = 0; i < num_featureMaps * size2 * size3; i++)
		{
			for (int j = 0; j < num_weights; j++)
			{
				lossInput[i / (size2 * size3)][i / size3 % size2][i % size3] += weights[i][j] * dLdt[j];
				weights[i][j] -= learn_rate * last_inputVector[i] * dLdt[j];
			}
		}
		
		delete[] dLdt;
		return lossInput;
	}
};

int main() 
{
	try
	{
		float x[42000][784];
		int y[42000];

		string line_v[785];

		ifstream myFile("train.txt");
		if (myFile.is_open())
		{
			int lineNum = 0;
			string line;
			while (getline(myFile, line))
			{
				istringstream ss(line);
				string token;
				int i = 0;
				while (getline(ss, token, '\t'))
				{
					int digit = stoi(token, nullptr);
					if (i == 0)
						y[lineNum] = digit;
					else
						x[lineNum][i - 1] = static_cast<float>(digit) / static_cast<float>(255);

					i++;
				}
				lineNum++;
			}
			myFile.close();
		}

		const int batchSize = 10;
		const int imageSize = 28;
		const int convLayers = 8;
		const int poolDimensions = 2;

		vector<vector<vector<float>>> x_batch(batchSize, vector<vector<float>>(imageSize, vector<float>(imageSize)));
		vector<int> y_batch(batchSize);

		Conv5x5 conv(convLayers, batchSize, imageSize, imageSize);
		MaxPool pool(poolDimensions, poolDimensions, convLayers * batchSize, imageSize - 2, imageSize - 2);
		FullyConnectedLayer conn(batchSize, (imageSize - 2) / 2, (imageSize - 2) / 2);

		const float learnRate = 0.001f;
		const float firstMomentum = 0.9f;
		const float secondMomentum = 0.999f;

		for (int i = 0; i < 100; i++) //TODO only 100?
		{
			int randIndex = rand() % (42000 - batchSize);
			for (unsigned j = 0; j < batchSize; j++)
			{
				for (int k = 0; k < 784; k++)
					x_batch[j][k / imageSize][k % imageSize] = x[j + randIndex][k];

				y_batch[j] = y[j + randIndex];
			}

			vector<vector<vector<float>>> help = conv.forward(x_batch);
			help = pool.forward(help);
			vector<float> res = conn.forward(help);

			help = conn.backprop(res, learnRate); //TODO change res?
			help = pool.backprop(help, learnRate);
			conv.backprop(help, learnRate);

			float loss = 0; //Welche Einheit hat der loss? In der Ausgabe entsprechend angeben
			int correct = 0;

			//TODO calculate loss and correct guesses

			if ((i + 1) % 100 == 0)
				cout << "Step " << i + 1 << " Average Loss " << loss << " Accuracy " << correct << "\n";
		}

		return 0;
	}
	catch(const exception&)
	{
		return -1;
	}
}
