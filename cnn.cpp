/*
 * cnn.cpp
 *
 *  Created on: 03.05.2021
 *      Author: Stefan, Hannah 
 */

#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <random>

using namespace std;
float x[42000][784];
float y[42000];

const int batchSize = 10;
float x_batch[batchSize][28][28];
float y_batch[batchSize];

class Conv3x3
{
public:
	int num_filters;
	int size1, size2, size3;
	vector<vector<vector<float>>> filters;
	vector<float> biases;

	vector<vector<vector<float>>> last_input;

	Conv3x3(int n, int s1, int s2, int s3)
	{
		num_filters = n;
		size1 = s1;
		size2 = s2;
		size3 = s3;
		filters.resize(3, vector<vector<float>>(3, vector<float>(num_filters)));
		biases.resize(num_filters, 0.0);

		normal_distribution<float> distribution(0.0, 1.0);
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				for (int k = 0; k < num_filters; k++)
				{
					default_random_engine generator;
					filters[i][j][k] = distribution(generator) / 9;
				}
			}
		}
	}

	vector<vector<vector<float>>> forward(vector<vector<vector<float>>> input)
	{
		vector<vector<vector<float>>> output(size1 * num_filters, vector<vector<float>>(size2 - 2, vector<float>(size3 - 2)));
		for (int i = 0; i < size2 - 2; i++)
		{
			//per region
			for (int j = 0; j < size3 - 2; j++)
			{
				// per region

				for (int k = 0; k < num_filters; k++)
				{
					//per filter
					for (int l = 0; l < size1; l++)
					{
						//per passed representation
						output[l * num_filters + k][i][j] = 0;
						output[l * num_filters + k][i][j] = biases[k];
						
						//set output at i j for the input representation l when filter k is applied
						//matrix multiplication and summation
						for (int m = 0; m < 3; m++)
							for (int n = 0; n < 3; n++)
								output[l * num_filters + k][i][j] += input[l][i + m][j + n] * filters[m][n][k];
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
				for (int k = 0; k < num_filters; k++)
				{
					//per filter
					for (int l = 0; l < size1; l++)
					{
						//per passed representation
						//matrix multiplication and summation
						for (int m = 0; m < 3; m++)
						{
							for (int n = 0; n < 3; n++)
							{
								filterGradient[m][n][k] += lossGradient[l * num_filters + k][i][j] * last_input[l][i + m][j + n];
								lossInput[l][i + m][j + n] += lossGradient[l * num_filters + k][i][j] * filters[m][n][k];
							}
						}

						filterBias[k] += lossGradient[l * num_filters + k][i][j];
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

class MaxPool2
{
public:
	int size1, size2, size3;
	int window, stride;

	vector<vector<vector<float>>> last_input;

	MaxPool2(int w, int s, int s1, int s2, int s3)
	{
		window = w;
		stride = s;
		size1 = s1;
		size2 = s2;
		size3 = s3;
	}

	vector<vector<vector<float>>> forward(vector<vector<vector<float>>> input)
	{
		vector<vector<vector<float>>> output(size1, vector<vector<float>>(((size2 - window) / stride) + 1,
			vector<float>(((size3 - window) / stride) + 1)));
		for (int i = 0; i < size2 - window; i += stride)
		{
			//per region
			for (int j = 0; j < size3 - window; j += stride)
			{
				// per region

				for (int l = 0; l < size1; l++)
				{
					//per passed representation
					//matrix max pooling
					float max = input[l][i][j];
					for (int m = 0; m < window; m++)
					{
						for (int n = 0; n < window; n++)
						{
							if (max < input[l][i + m][j + n])
								max = input[l][i + m][j + n];
						}
						
						output[l][i / stride][j / stride] = max;
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
				for (int l = 0; l < size1; l++)
				{
					//per passed representation
					//matrix max pooling
					float max = last_input[l][i][j];
					int indexX = 0;
					int indexY = 0;
					for (int m = 0; m < window; m++)
					{
						for (int n = 0; n < window; n++)
						{
							if (max < last_input[l][i + m][j + n])
							{
								max = last_input[l][i + m][j + n];
								indexX = m;
								indexY = n;
							}
						}
						
						lossInput[l][i + indexX][j + indexY] = lossGradient[l][i][j];
					}
				}
			}
		}

		return lossInput;
	}
};

class FullyConnectedLayer
{
public:
	int num_featuremaps;	//Number of feature maps the convolutional Layers generate
	int size2, size3;		//Dimensions of the feature maps
	static const int num_weights = 10; //Number of 
	vector<vector<float>> weights;
	vector<float> biases;

	vector<float> last_inputVector;
	vector<float> last_totals;
	float last_sum = 0.0;

	FullyConnectedLayer(int s1, int s2, int s3) {
		num_featuremaps = s1;
		size2 = s2;
		size3 = s3;
		weights.resize(num_featuremaps * size2 * size3, vector<float>(num_weights));
		biases.resize(num_weights, 0.0);

		normal_distribution<float> distribution(0.0, 1.0);
		for (int i = 0; i < num_featuremaps * size2 * size3; i++)
		{
			for (int j = 0; j < num_weights; j++)
			{
				default_random_engine generator;
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
		vector<float> inputVector(num_featuremaps * size2 * size3);
		for (int i = 0; i < num_featuremaps; i++)
			for (int j = 0; j < size2; j++)
				for (int k = 0; k < size3; k++)
					inputVector[i * size2 * size3 + j * size3 + k] = input[i][j][k];

		for (int i = 0; i < num_featuremaps * size2 * size3; i++) //per feature
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
		vector<vector<vector<float>>> lossInput(num_featuremaps, vector<vector<float>>(size2, vector<float>(size3, 0.0)));

		int index = -1;
		for (int i = 0; i < num_weights; i++)
		{
			if (lossGradient[i] != 0)
				index = i;
		}
		const float gradient = lossGradient[index];

		float doutdt[num_weights];
		for (int i = 0; i < num_weights; i++)
		{
			doutdt[i] = - last_totals[index] * last_totals[i] / (last_sum * last_sum);
		}
		doutdt[index] = last_totals[index] * last_sum - last_totals[index] / (last_sum * last_sum);

		float* dLdt = new float[num_weights];
		for (int i = 0; i < num_weights; i++)
			dLdt[i] = gradient * doutdt[i];

		for (int i = 0; i < num_featuremaps * size2 * size3; i++)
			for (int j = 0; j < num_weights; j++)
				lossInput[i / (size2 * size3)][i / size3 % size2][i % size3] += weights[i][j] * dLdt[j];

		for (int i = 0; i < num_featuremaps * size2 * size3; i++)
			for (int j = 0; j < num_weights; j++)
				weights[i][j] -= learn_rate * last_inputVector[i] * dLdt[j];

		for (int i = 0; i < num_weights; i++)
			biases[i] -= learn_rate * dLdt[i];

		delete[] dLdt;
		return lossInput;
	}
};

int main()
{
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
				float digit = strtof(token.c_str(), nullptr);
				if (i == 0)
					y[lineNum] = digit;
				else
					x[lineNum][i - 1] = digit / 255;
				
				i++;
			}
			lineNum++;
		}
		myFile.close();
	}

	for (int i = 0; i < 100; i++)
	{
		int randIndex = rand() % (42000 - batchSize);
		for (unsigned j = 0; j < batchSize; j++)
		{
			for (int k = 0; k < 784; k++)
			{
				x_batch[k / 28][k % 28][j] = x[j + randIndex][k];
			}
			y_batch[j] = y[j + randIndex];
		}
	}

	return 0;
}
