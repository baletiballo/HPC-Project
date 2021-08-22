#ifndef PARAMETER_H_
#define PARAMETER_H_

/*
    Hier sind alle Parameter, die für einen gesamten Trainingsprozess konstant sind.
    Durch das Einbinden dieser Datei müssen die Daten nicht an (fast) alle Konstruktoren übergeben werden.
    Um einen Schwung von Läufen, mit leicht verschiedenen Parametern zu machen, sollte das evtl. ein Parameter struct werden, 
    das man übergeben kann.
*/

#include <cmath>

const int baseSizeX = 28; //Anzahl Pixel in der X-Richtung, der originalen Trainingsdaten
const int baseSizeY = 28; //        -||-        Y-Richtung,              -||-        
const int batchSize = 32; //Anzahl Bilder pro Batch
const int infaltionFactor = 2; //Faktor, um den jedes Bild hochskaliert wird. Also 1 px -> Block mit Kantenlänge infationFactor
const int imageSizeX = ((baseSizeX - 1) * infaltionFactor) + 1 ; //Anzahl Pixel in der X-Richtung, der Trainingsdaten
const int imageSizeY = ((baseSizeY - 1) * infaltionFactor) + 1 ; //Anzahl Pixel in der Y-Richtung, der Trainingsdaten
const int imagePixels = imageSizeX*imageSizeY; //Anzahl der Pixel eines unskalierten Bildes == Größe eines flachen Vektors eines Bildes
const int num_steps = 3001; //Anzahl an Batches
//Konstanten für ADAM, direkt die aus dem Paper
const float alpha = 0.001; //Lernrate
const float beta1 = 0.9; //Erstes Moment
const float beta2 = 0.999; //Zweites Moment
const float EPSILON = 1.0f * pow(10.0f, -8);

#endif /* CONV_H_ */
