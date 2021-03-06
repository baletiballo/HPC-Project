/*
 * train.cpp
 *
 *  Created on: 03.05.2021
 *      Author: Stefan Gebhart, Hannah Frisch, Silas Kuder
 */

//Werden in CNN.cpp eingebunden, hier aber auch genutzt
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <ctime> 
#include <iomanip> 

#include "parameter.h"
#include "cnn.h"

using namespace std;

/*
 Ließt die Trainingsdaten ein, und skaliert sie, um aussagekräftigere Inputs bieten zu können
 */
void read_scale_trainingData(string filename, float training_images [num_trainingData] [imageSizeX] [imageSizeY], int_fast8_t correct_lables[num_trainingData], int factor);
void scale_trainingData(vector<vector<vector<float>>> tmp_images, float training_images [num_trainingData] [imageSizeX] [imageSizeY], int factor); //skaliert die Tainingsdaten
//Die ursprüngliche main(). So können mehrere ähnliche Trainingsläufe durchgeführt werden, ohne dass sie alle einzeln angestoßen werden müssen
void train();

float endLoss; //Gesamter Loss der letzten 10 Batches
int_fast16_t endCorr; //Gesamtanzahl korrekt geratener Labels der letzten 10 Batches
chrono::duration<double> totalTime; //Gesamtzeit des Trainings


//Alle Trainingsdaten, als Vektor von Graustufen Matrizen
float training_images [num_trainingData] [imageSizeX] [imageSizeY];
int_fast8_t correct_lables [num_trainingData];

//Alle Bilder eines Batches, als Vektor von Graustufen Matrizen
float (*batch_images) [imageSizeX] [imageSizeY];
//Alle Lables eines Batches, als Vektor von Ganzzahlen
int_fast8_t (*batch_lables);

/////////////////////////////////////////

int main() {
	fstream log;
	double avgTime = 0.0;
	log.open("Testlog.txt", std::ios_base::app);
	if (!log.is_open())
		return -1;
	read_scale_trainingData("train.txt", training_images, correct_lables, infaltionFactor);
	/////////////////////////

	for (int i = 1; i <= 1; i++) { //Bündellauf mit leicht unterschiedlichen Parametern
		//Parameter setzten
		

		log << "------------------------------------------" << endl;
		//log << "| " << setw(35) << "Batchsize = " << setw(3) << batchSize << " |" << endl;
		log << "| " << setw(35) << "Anzahl an Threads = " << setw(3) << threads << " |" << endl;
		//log << "| " << setw(38) << "Update parallel" << " |" << endl;
		//log << "| " << setw(38) << "Inflationfaktor = 3" << " |" << endl;
		log << "------------------------------------------" << endl;

		std::cout << "| " << setw(35) << "Anzahl an Threads = " << setw(3) << threads << " |" << endl;
		for (int j = 0; j < num_trainings_cycles; j++) { //Mehrere Durchläufe mit denselben Parametern, um Konsistenz zu erhöhen		
			
 			train();
			if(detailedLog){
				if (num_trainings_cycles != 1) log << "Trainingsdurchlauf " << j << ":" << endl;
				log << "Durchschnittlicher Loss in den letzten 10 Batches:" << endLoss / (float)(10 * batchSize)
					<< "\t Durchschnittliche Praezision in den letzten 10 Batches: " << (float)endCorr / (10 * batchSize) << endl;
				log << totalTime.count()<< " Sekunden" << endl << endl;
			}
			
			endLoss = 0.0;
			endCorr = 0;
			avgTime += totalTime.count();
		}

		if (num_trainings_cycles != 1){
			avgTime /= num_trainings_cycles;
			log << "Durchschnittlich: " << avgTime << " Sekunden" << endl << endl;
			avgTime = 0.0;
		}
	}
	std::cout << "Training finished." << endl;
	log << endl;

	//Beenden des Threadpools
	sem.set(0);
	endThreads();
	sem.P(threads);
	return 0;
}

//Die ursprüngliche main(). So können mehrere ähnliche Trainingsläufe durchgeführt werden, ohne dass sie alle einzeln angestoßen werden müssen
void train() {

	endCorr = 0;
	endLoss = 0;

	try {
		CNN cnn;		//Das benutzte Netzwerk.
		pool.setCNN(cnn);

		//std::cout << "Beginn des Trainings\n";
		auto training_startTime = chrono::system_clock::now(); // Interner Timer um die Laufzeit zu messen

		for (int i = 0; i < num_steps; i++) 
		{			
			/* Vorbereiten des Trainingsbatches */
			int randIndex = rand() % (num_trainingData - batchSize);
			batch_images =  &training_images[randIndex];
			batch_lables =  &correct_lables[randIndex];

			tuple<float, int> res = cnn.learn(batch_images, batch_lables);

			float loss = get<0>(res) / batchSize;
			float prez = (float)get<1>(res) / (batchSize);

			
			//cout<<"Durchschnittlicher Loss: " << get<0>(res) / (float)(batchSize)
			//		 << "\t Durchschnittliche Praezision: " << (float)get<1>(res) / (batchSize) << endl;
			
			if(num_steps - i <=10 ){
				endLoss += get<0>(res);
				endCorr += get<1>(res);
			}
			
			
		}

		auto training_endTime = chrono::system_clock::now();
		totalTime = training_endTime - training_startTime;
		//endThreads();
	} catch (const exception&) {
		//endThreads();
		std::cout << "Fehler => Abbruch" << endl;
	}
}

void read_scale_trainingData(string filename, float training_images [num_trainingData] [imageSizeX] [imageSizeY], int_fast8_t correct_lables [num_trainingData], int factor) {
	ifstream myFile(filename);
	if (myFile.is_open()) {
		vector<vector<vector<float>>> tmp_images (num_trainingData, vector<vector<float>> (baseSizeX, vector<float> (baseSizeY))); //Die originalen (nicht skalierten) Trainingsdaten
		//cout << "Lese Trainingsdaten ein;\n";
		string line;
		for(int lineNum = 0; lineNum < num_trainingData; lineNum++){
			getline(myFile, line);
			istringstream ss(line);
			string token;
			int i = 0;
			while (getline(ss, token, '\t')) {
				int digit = stoi(token, nullptr);
				if (i == 0)			//erste Zahl jeder Zeile ist das lable
					correct_lables[lineNum] = digit;
				else				//der Rest das Graustufenbild
					tmp_images [lineNum] [(i - 1)/baseSizeX] [(i - 1)%baseSizeX] = static_cast<float>(digit) / static_cast<float>(255);
				i++;
			}

			lineNum++;
		}
		myFile.close();
		scale_trainingData(tmp_images, training_images, factor);
	}
}

void scale_trainingData(vector<vector<vector<float>>> tmp_images, float training_images [num_trainingData] [imageSizeX] [imageSizeY], int factor) {
	//Bilineare Interpolation
	for (int image = 0; image < num_trainingData; image++) {
		for (unsigned pixelX = 0; pixelX < imageSizeX; pixelX++) {
			for (unsigned pixelY = 0; pixelY < imageSizeY; pixelY++) {
				if (pixelX % factor == 0) {
					if (pixelY % factor == 0) {
						training_images[image][pixelX][pixelY] = tmp_images[image][pixelX / factor][pixelY / factor];
					} else {
						training_images[image][pixelX][pixelY] = (tmp_images[image][pixelX / factor][pixelY / factor] * (factor - (pixelY % factor))
								+ tmp_images[image][pixelX / factor][(pixelY / factor) + 1] * (pixelY % factor)) / factor;
					}
				} else {
					if (pixelY % factor == 0) {
						training_images[image][pixelX][pixelY] = (tmp_images[image][pixelX / factor][pixelY / factor] * (factor - (pixelX % factor))
								+ tmp_images[image][(pixelX / factor) + 1][pixelY / factor] * (pixelX % factor)) / factor;
					} else {
						training_images[image][pixelX][pixelY] = (tmp_images[image][pixelX / factor][pixelY / factor] * (factor - (pixelX % factor)) * (factor - (pixelY % factor))
								+ tmp_images[image][(pixelX / factor) + 1][pixelY / factor] * (pixelX % factor) * (factor - (pixelY % factor))
								+ tmp_images[image][pixelX / factor][(pixelY / factor) + 1] * (factor - (pixelX % factor)) * (pixelY % factor)
								+ tmp_images[image][(pixelX / factor) + 1][(pixelY / factor) + 1] * (pixelX % factor) * (pixelY % factor)) / (factor * factor);
					}
				}
			}
		}
	}
}
