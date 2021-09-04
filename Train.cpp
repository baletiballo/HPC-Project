/*
 * train.cpp
 *
 *  Created on: 03.05.2021
 *      Author: Stefan, Hannah, Silas
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
	log.open("Testlog.txt", std::ios_base::out);
	if (!log.is_open())
		return -1;
	read_scale_trainingData("train.txt", training_images, correct_lables, infaltionFactor);
	/////////////////////////

	for (int i = 1; i <= 1; i++) { //Bündellauf mit leicht unterschiedlichen Parametern
		//Parameter setzten
		log << "------------------------------------------" << endl;
		log << "| " << setw(38) << "Vektoren durch Arrays ersetzt" << " |" << endl;
		log << "| " << setw(38) << "Einige Loops unrolled" << " |" << endl;
		log << "------------------------------------------" << endl;

		for (int j = 0; j < num_trainings_cycles; j++) { //Mehrere Durchläufe mit denselben Parametern, um Konsistenz zu erhöhen		
			log << "Trainingsdurchlauf " << j << ":" << endl;
 			train();
			log << "Durchschnittlicher Loss in den letzten 10 Batches:" << endLoss / (float)(10 * batchSize)
			 << "\t Durchschnittliche Praezision in den letzten 10 Batches: " << (float)endCorr / (10 * batchSize) << endl;
			log << (int) (totalTime.count() / 60) << " Minuten " << (int) (totalTime.count()) % 60 << " Sekunden" << endl;
			endLoss = 0.0;
			endCorr = 0;
			avgTime += totalTime.count();
			log << endl;
		}

		avgTime /= num_trainings_cycles;
		log << "Durchschnittlich: " << (int) (avgTime / 60) << " Minuten " << (int) (avgTime) % 60 << " Sekunden" << endl << endl;
		avgTime = 0.0;
	}

	log << endl;
	std::cout << "Training finished." << endl;
}

//Die ursprüngliche main(). So können mehrere ähnliche Trainingsläufe durchgeführt werden, ohne dass sie alle einzeln angestoßen werden müssen
void train() {

	endCorr = 0;
	endLoss = 0;

	try {
		CNN cnn;		//Das benutzte Netzwerk.

		//std::cout << "Beginn des Trainings\n";
		auto training_startTime = chrono::system_clock::now(); // Interner Timer um die Laufzeit zu messen

		for (int i = 0; i < num_steps; i++) 
		{			
			/* Vorbereiten des Trainingsbatches */
			int randIndex = rand() % (num_trainingData - batchSize);
			batch_images =  &training_images[randIndex];
			batch_lables =  &correct_lables[randIndex];

			tuple<float, int_fast8_t> res = cnn.learn(batch_images, batch_lables);

			if (num_steps - i <= 10) {
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

//an image scaled by this should have sizes:
//((image.size() - 1) * factor) + 1 by (((image[0].size() - 1) * factor) + 1))
//this is because we want to add factor-1 new pixels between each two original pixels. This gives us image.size()-1 blocks with factor pixels each and one single lone pixel at the end.
//sooo make sure training_images[image] has those sizes
void scale_image_bilinear_interpolation(float image [baseSizeX] [baseSizeY], float newImage [imageSizeX] [imageSizeY], int factor) {}

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
	/*
	fstream log;
	log.open("Testlog.txt", std::ios_base::app);
	if (!log.is_open())
		return;
	for (size_t x = 0; x < training_images[image].size(); x++)
	{
		for (int y = 0; y < training_images[image][0].size(); y++){
			if(training_images[image][x][y] > 0.1) 
				log << "00";
			else  if(training_images[image][x][y] > 0.01)
				log << "++";
			else
				log << "..";
		}
		log << endl;
	}
	log << endl << endl;
	*/
	
	/* //Reines Aufblasen des Bildes
	for (int image = 0; image < num_trainingData; image++) {
		for (int pixel = 0; pixel < imagePixels; pixel++) {
			float pixelValue = tmp_images[image][pixel];
			for (int xShift = 0; xShift < factor; xShift++) {
				for (int yShift = 0; yShift < factor; yShift++) {
					training_images[image][factor * (pixel / 28) + xShift][factor * (pixel % 28) + yShift] = pixelValue;
				}
			}
		}
	}
	*/
}
