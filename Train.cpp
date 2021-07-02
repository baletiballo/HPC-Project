/*
 * cnn.cpp
 *
 *  Created on: 03.05.2021
 *      Author: Stefan, Hannah, Silas
 */

//Werden in CNN.cpp eingebunden, hier aber auch genutzt
//#include <tuple>
//#include <vector>

#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <ctime> 

#include "cnn.cpp"
#include "parameter.h"


using namespace std;

/*
Ließt die Trainingsdaten ein, und skaliert sie, um aussagekräftigere Inputs bieten zu können

*/ 
void read_scale_trainingData(string filename,vector<vector<vector<float>>> &training_images, vector<int_fast8_t> &correct_lables, int factor);
void scale_trainingData (vector<vector<float>> tmp_images, vector<vector<vector<float>>> &training_images,int factor);//skaliert die Tainingsdaten

float endLoss; //Gesamter Loss der letzten 10 Batches
int_fast16_t endCorr; //Gesamtanzahl korrekt geratener Labels der letzten 10 Batches

//Alle Trainingsdaten, als Vektor von Graustufen Matrizen
vector<vector<vector<float>>> training_images(42000, vector<vector<float>>(imageSize, vector<float>(imageSize)));
vector<int_fast8_t> correct_lables(42000);

//Alle Bilder eines Batches, als Vektor von Graustufen Matrizen
vector<vector<vector<float>>> batch_images(batchSize, vector<vector<float>>(imageSize, vector<float>(imageSize))); 
//Alle Lables eines Batches, als Vektor von Ganzzahlen
vector<int_fast8_t> batch_lables(batchSize);

int main() {
	try {
		/* Einlesen der Trainingsdaten*/
		read_scale_trainingData("train.txt", training_images, correct_lables, infaltionFactor);

		CNN cnn(batchSize);//Das benutzte Netzwerk. Topologieänderungen bitte in der Klasse CNN		

		cout << "Beginn des Trainings\n";
		auto training_startTime = chrono::system_clock::now(); // Interner Timer um die Laufzeit zu messen

		for (int i = 0; i < num_steps; i++) {

			/* Vorbereiten des Trainingsbatches */
			int randIndex = rand() % (42000 - batchSize);
			for (unsigned j = 0; j < batchSize; j++) { //erstelle einen zufälligen Batch für das Training	
				batch_images[j] = training_images[j + randIndex];
				batch_lables[j] = correct_lables[j + randIndex];
			}

			tuple<float, int_fast8_t> res = cnn.learn(batch_images, batch_lables);

			float loss = get<0>(res);
			int_fast8_t correct = get<1>(res);

			if(i % 500 == 0){//Zwischenupdates. Nur alle paar hundert Baches, um Konsole übersichtlich zu halten
				cout << "Batch " << i << " \t Average Loss " << loss / batchSize << "\t Accuracy " << (int)correct <<"/"<< batchSize << "\n";
			}

			if (num_steps - i <= 10) {
				endLoss += loss;
				endCorr += correct;
			}
		}

		auto training_endTime = chrono::system_clock::now();
		chrono::duration<double> totalTime = training_endTime - training_startTime;
		cout << "Total time: " << (int) (totalTime.count() / 60) << " minutes " << (int) (totalTime.count()) % 60 << " seconds\n";
		cout << "Average loss in last " << batchSize * 10 << " tries:" << endLoss / (float)(10 * batchSize) << "\t Average accuracy in last 10 batches: "
				<< (float)endCorr / (10 * batchSize) << "\n";
		//endThreads();

		return 0;
	} catch (const exception&) {
		//endThreads();
		cout <<"Fehler => Abbruch\n";
		return -1;
	}
}

void read_scale_trainingData(string filename, vector<vector<vector<float>>> &training_images, vector<int_fast8_t> &correct_lables, int factor) {
	ifstream myFile(filename);
	if (myFile.is_open()) {
		vector<vector<float>> tmp_images(42000, vector<float>(imagePixels)); //Die originalen (nicht skalierten) Trainingsdaten
		cout << "Lese Trainingsdaten ein\n";
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
					tmp_images[lineNum][i - 1] = static_cast<float>(digit) / static_cast<float>(255);
				i++;
			}

			lineNum++;
		}
		myFile.close();
		scale_trainingData(tmp_images, training_images, factor);
	}
}

void scale_trainingData (vector<vector<float>> tmp_images, vector<vector<vector<float>>> &training_images,int factor){
	for (int image = 0; image < 42000; image++){
		for(int pixel = 0; pixel < imagePixels; pixel++){
			float pixelValue = tmp_images[image][pixel];
			for(int xShift = 0; xShift < factor; xShift++){
				for(int yShift = 0; yShift < factor; yShift++){
					training_images[image][factor * (pixel/28) + xShift][factor * (pixel%28) +yShift] = pixelValue;
				}
			}
		}
	}
}