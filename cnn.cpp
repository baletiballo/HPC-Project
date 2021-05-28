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
#include <float.h>
#include <chrono>
#include <ctime> 

using namespace std;

class Conv5x5 {
public:
	int num_filters;
	int size1, size2, size3;
	const int conv_size = 5;
	vector<vector<vector<float>>> filters;
	vector<float> biases;

	vector<vector<vector<float>>> last_input;

	Conv5x5(int n, int s1, int s2, int s3) {
		num_filters = n;
		size1 = s1;
		size2 = s2;
		size3 = s3;
		filters.resize(conv_size, vector<vector<float>>(conv_size, vector<float>(num_filters)));
		biases.resize(num_filters, 0.0);

		normal_distribution<float> distribution(0.0, 1.0);
		for (int i = 0; i < conv_size; i++) {
			for (int j = 0; j < conv_size; j++) {
				for (int cur_filter = 0; cur_filter < num_filters; cur_filter++) {
					random_device dev;
					default_random_engine generator(dev());
					filters[i][j][cur_filter] = distribution(generator) / 9;
				}
			}
		}
	}

	vector<vector<vector<float>>> forward(vector<vector<vector<float>>> &input) {
		const int numWindows = size2 - conv_size + 1;
		vector<vector<vector<float>>> output(size1 * num_filters, vector<vector<float>>(numWindows, vector<float>(size3 - (conv_size - 1))));
		for (int i = 0; i < numWindows; i++) {
			//per region
			for (int j = 0; j < numWindows; j++) {
				// per region
				for (int cur_filter = 0; cur_filter < num_filters; cur_filter++) {
					//per filter
					for (int cur_featureMap = 0; cur_featureMap < size1; cur_featureMap++) {
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

	vector<vector<vector<float>>> backprop(vector<vector<vector<float>>> &lossGradient, float learn_rate) {
		vector<vector<vector<float>>> filterGradient(conv_size, vector<vector<float>>(conv_size, vector<float>(num_filters, 0.0)));
		vector<float> filterBias(num_filters, 0.0);
		vector<vector<vector<float>>> lossInput(size1, vector<vector<float>>(size2, vector<float>(size3, 0.0)));

		for (int i = 0; i < size2 - (conv_size-1); i++) {
			//per region
			for (int j = 0; j < size3 - (conv_size-1); j++) {
				// per region
				for (int cur_filter = 0; cur_filter < num_filters; cur_filter++) {
					//per filter
					for (int cur_featureMap = 0; cur_featureMap < size1; cur_featureMap++) {
						//per passed representation
						//matrix multiplication and summation
						for (int m = 0; m < conv_size; m++) {
							for (int n = 0; n < conv_size; n++) {
								filterGradient[m][n][cur_filter] += lossGradient[cur_featureMap * num_filters + cur_filter][i][j]
										* last_input[cur_featureMap][i + m][j + n];
								lossInput[cur_featureMap][i + m][j + n] += lossGradient[cur_featureMap * num_filters + cur_filter][i][j]
										* filters[m][n][cur_filter];
							}
						}

						filterBias[cur_filter] += lossGradient[cur_featureMap * num_filters + cur_filter][i][j];
					}
				}
			}
		}

		for (int i = 0; i < num_filters; i++)
			biases[i] -= learn_rate * filterBias[i];

		for (int i = 0; i < conv_size; i++)
			for (int j = 0; j < conv_size; j++)
				for (int k = 0; k < num_filters; k++)
					filters[i][j][k] -= learn_rate * filterGradient[i][j][k];

		return lossInput;
	}
};

class MaxPool {
public:
	int size1, size2, size3;
	int window, stride;

	vector<vector<vector<float>>> last_input;

	MaxPool(int w, int s, int s1, int s2, int s3) {
		window = w;
		stride = s;
		size1 = s1;
		size2 = s2;
		size3 = s3;
	}

	vector<vector<vector<float>>> forward(vector<vector<vector<float>>> &input) {
		vector<vector<vector<float>> > output(size1, vector<vector<float>>((size2 - window) / stride + 1, vector<float>((size3 - window) / stride + 1)));
		for (int i = 0; i < size2 - window; i += stride) {
			//per region
			for (int j = 0; j < size3 - window; j += stride) {
				// per region
				for (int cur_featureMap = 0; cur_featureMap < size1; cur_featureMap++) {
					//per passed representation
					//matrix max pooling
					float max = input[cur_featureMap][i][j];
					for (int m = 0; m < window; m++) {
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

	vector<vector<vector<float>>> backprop(vector<vector<vector<float>>> &lossGradient) {
		vector<vector<vector<float>>> lossInput(size1, vector<vector<float>>(size2, vector<float>(size3, 0.0)));

		for (int i = 0; i < size2 - window; i += stride) {
			//per region
			for (int j = 0; j < size3 - window; j += stride) {
				// per region
				for (int cur_featureMap = 0; cur_featureMap < size1; cur_featureMap++) {
					//per passed representation
					//matrix max pooling
					float max = last_input[cur_featureMap][i][j];
					int indexX = 0;
					int indexY = 0;
					for (int m = 0; m < window; m++) {
						for (int n = 0; n < window; n++) {
							if (max < last_input[cur_featureMap][i + m][j + n]) {
								max = last_input[cur_featureMap][i + m][j + n];
								indexX = m;
								indexY = n;
							}
						}
					}

					//set only the lossInput of the "pixel" max pool kept
					lossInput[cur_featureMap][i + indexX][j + indexY] = lossGradient[cur_featureMap][i / stride][j / stride];
				}
			}
		}

		return lossInput;
	}
};

class FullyConnectedLayer {
public:
	int num_featureMaps; //Number of feature maps the convolutional Layers generate
	int size2, size3; //Dimensions of the feature maps
	static const int num_weights = 10; //Anzahl der Ausgabeklassen
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
		for (int i = 0; i < num_featureMaps * size2 * size3; i++) {
			for (int j = 0; j < num_weights; j++) {
				random_device dev;
				default_random_engine generator(dev());
				weights[i][j] = distribution(generator) / 9;
			}
		}

		last_totals.resize(num_weights);
	}

	vector<float> forward(vector<vector<vector<float>>> &input) {
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
		for (int i = 0; i < num_weights; i++) {
			output[i] = exp(output[i]);
			last_totals[i] = output[i];
			total += output[i];
		}
		last_sum = total;

		//normalize
		for (int i = 0; i < num_weights; i++) {
			output[i] = output[i] / total;
		}
		return output;
	}

	vector<vector<vector<float>>> backprop(vector<float> lossGradient, float learn_rate) {
		vector<vector<vector<float>>> lossInput(num_featureMaps, vector<vector<float>>(size2, vector<float>(size3, 0.0)));

		int index = -1;
		for (int i = 0; i < num_weights; i++) {
			if (lossGradient[i] !=0)
				index = i;
		}

		const float gradient = lossGradient[index];

		float dOutDt[num_weights];
		for (int i = 0; i < num_weights; i++)
			dOutDt[i] = -last_totals[index] * last_totals[i] / (last_sum * last_sum);

		dOutDt[index] = last_totals[index] * (last_sum - last_totals[index]) / (last_sum * last_sum);

		const auto dLdt = new float[num_weights];
		for (int i = 0; i < num_weights; i++) {
			dLdt[i] = gradient * dOutDt[i];
			biases[i] -= learn_rate * dLdt[i];
		}

		for (int i = 0; i < num_featureMaps * size2 * size3; i++) {
			for (int j = 0; j < num_weights; j++) {
				lossInput[i / (size2 * size3)][i / size3 % size2][i % size3] += weights[i][j] * dLdt[j];
				weights[i][j] -= learn_rate * last_inputVector[i] * dLdt[j];
			}
		}

		delete[] dLdt;
		return lossInput;
	}
};

class CNN {
public:
	static const int sizeX=28;
	static const int sizeY=28;
	static const int num_conv_layers=2;
	const int conv_layers_num_filters[num_conv_layers]={8,8};
	const int pool_layers_window[num_conv_layers]={2,2};
	const int pool_layers_stride[num_conv_layers]={2,2};

	vector<Conv5x5> conv_layers;
	vector<MaxPool> pooling_layers;
	FullyConnectedLayer *connected_layer;

	CNN() {
		int currX=sizeX;
		int currY=sizeY;
		int images=1;
		for(unsigned i=0;i<num_conv_layers;i++) {
			conv_layers.push_back(Conv5x5(conv_layers_num_filters[i],images,currX,currY));
			currX-=4;
			currY-=4;
			images*=conv_layers_num_filters[i];
			pooling_layers.push_back(MaxPool(pool_layers_window[i], pool_layers_stride[i], images, currX, currY));
			currX=(currX - pool_layers_window[i]) / pool_layers_stride[i] + 1;
			currY=(currY - pool_layers_window[i]) / pool_layers_stride[i] + 1;
		}
		connected_layer=new FullyConnectedLayer(images, currX, currY);
	}

	vector<float> forward(vector<vector<vector<float>> > &image) {
		vector<vector<vector<float>> > help=conv_layers[0].forward(image);
		help=pooling_layers[0].forward(help);
		for(int i=1;i<num_conv_layers;i++) {
			help=conv_layers[i].forward(help);
			help=pooling_layers[i].forward(help);
		}
		return (*connected_layer).forward(help);
	}

	void backprop(vector<float> &res, float lr) {
		vector<vector<vector<float>> > help= (*connected_layer).backprop(res, lr);
		for(int i=num_conv_layers-1;i>-1;i--) {
			help = pooling_layers[i].backprop(help);
			help = conv_layers[i].backprop(help, lr);
		}
	}
};

int main() {
	try {
		vector<vector<float>> training_images(42000, vector<float>(784));
		vector<int> correct_lables(42000);

		/* Einlesen der Trainingsdaten*/
		read_trainingData("train.txt",training_images,correct_lables);

		/*Vorbereiten des Netzwerks für das Training*/
		const int batchSize = 1000;
		const int imageSize = 28;
		const int num_steps = 10;
		const float learnRate = 0.001;

		vector<vector<vector<float>>> batch_images(batchSize, vector<vector<float>>(imageSize, vector<float>(imageSize)));
		vector<int> batch_lables(batchSize);
		CNN cnn; //Das benutzte Netzwerk. Topologieänderungen bitte in der Klasse CNN
		auto training_startTime = chrono::system_clock::now(); // Interner Timer um die Laufzeit zu messen

		for (int i = 0; i < num_steps; i++) {

			/* Vorberiten des Trainingsbatches */
			int randIndex = rand() % (42000 - batchSize);
			for (unsigned j = 0; j < batchSize; j++) { //erstelle einen zufälligen Batch für das Training
				for (int k = 0; k < 784; k++)//Reformatierung des flachen Vektors in Zeilen und Spalten
					batch_images[j][k / imageSize][k % imageSize] = training_images[j + randIndex][k]; 

				batch_lables[j] = correct_lables[j + randIndex];
			}

			float loss = 0; 
			float correct = 0;

			/* Tatsächliches Training */
			for (unsigned step = 0; step < batchSize; step++) {
				vector<vector<vector<float>> > image(1, vector<vector<float>>(imageSize, vector<float>(imageSize))); //Die Conv Layer generieren zusätzliche Bilder, daher ein 3D-Vektor
				image[0] = batch_images[step];
				vector<float> res = cnn.forward(image);

				loss += -log(res[batch_lables[step]]);

				/* Bestimme die Vorhersage des Netzwerks <=> Das Lable mit der höchsten Wahrscheinlichkeit */
				int predicted = 0;
				for (int k = 0; k < FullyConnectedLayer::num_weights; k++)
					if (res[k] >= res[predicted])
						predicted = k;
				if (predicted == batch_lables[step]) {
					correct += 1;
				} 

				for (int k = 0; k < FullyConnectedLayer::num_weights; k++)
					if (k == batch_lables[step])
						res[k] = -1 / res[k];
					else
						res[k] = 0;

				cnn.backprop(res,learnRate);
			}

			cout << "Batch " << i + 1 << " Average Loss " << loss / batchSize << " Accuracy " << correct / batchSize << "\n";
		}

		auto training_endTime = chrono::system_clock::now();
		chrono::duration<double> totalTime = training_endTime-training_startTime;
		cout << "Total time: " << (int)(totalTime.count()/60) << " minutes " << (int)(totalTime.count()) % 60 << " seconds\n";
		cout << "Average batch time: " << (totalTime.count()/ num_steps) << "seconds\n";
		return 0;
	} catch (const exception&) {
		return -1;
	}
}

int read_trainingData(string filename, vector<vector<float>> training_images, vector<int> correct_lables)
{
	ifstream myFile(filename);
		if (myFile.is_open()) {
			int lineNum = 0;
			string line;
			while (getline(myFile, line))
			{
				istringstream ss(line);
				string token;
				int i = 0;
				while (getline(ss, token, '\t')) {
					int digit = stoi(token, nullptr);
					if (i == 0)			//erste Zahl jeder Zeile ist das lable
						correct_lables[lineNum] = digit;
					else				//der Rest das Graustufenbild
						training_images[lineNum][i - 1] = static_cast<float>(digit) / static_cast<float>(255);

					i++;
				}

				lineNum++;
			}
			myFile.close();
		}

}

/*int adamGD(int batch, int n_c, int params, int cost) //TODO Die Datentypen der Eingangsparameter und der Rückgabe passen noch nicht
		{
	const int num_classes = 10; //Anzahl der Klassifikationsklassen (Hier 0 bis 9)
	const float lr = 0.01;		//Lernrate
	const float beta1 = 0.95;	//Erstes Moment
	const float beta2 = 0.99;	//Zweites Moment
	const int img_dim = 28;		//Größe der Bilder: 28x28 Pixel	

	//TODO Bisher nur ein Methodenstumpf
	return 0;
}

/*vector<float> forward(vector<vector<vector<float>> > &image) {
	vector<vector<vector<float>> > help = conv.forward(image);
	help = pool.forward(help);
	vector<float> res = conn.forward(help);
}

int adam(vector<vector<vector<float>>> &x_batch, vector<int> &y_batch) {
	const float lr = 0.01;		//Lernrate
	const float beta1 = 0.95;	//Erstes Moment
	const float beta2 = 0.99;	//Zweites Moment



	return 0;
}*/
