#ifndef PARAMETER_H_
#define PARAMETER_H_

/*
    Hier sind alle Parameter, die für einen gesamten Trainingsprozess konstant sind.
    Durch das Einbinden dieser Datei müssen die Daten nicht an (fast) alle Konstruktoren übergeben werden.
    
    Um einen Schwung von Läufen, mit leicht verschiedenen Parametern zu machen, sollte das evtl. ein Parameter struct werden, 
    das man übergeben kann.
*/

#include <cmath>

//Alle benötigten Klassen
class CNN;
class Conv;
class MaxPool;
class FullyConnectedLayer;
class JobQueue;
class ThreadPool;

//Parameter der Trainingsdaten
const int num_trainingData = 42000; //Anzahl an Trainingsdatensätzen (42000 bei MNist)
const int baseSizeX = 28; //Anzahl Pixel in der X-Richtung, der originalen Daten
const int baseSizeY = 28; //        -||-        Y-Richtung,          -||-        
const int batchSize = 32; //Anzahl Bilder pro Batch
const int infaltionFactor = 3; //Zwischen zwei Originalpixel werden (inflationFaktor-1) interpoliert für die Trainingsdaten
const int imageSizeX = ((baseSizeX - 1) * infaltionFactor) + 1 ; //DO NOT CHANGE: Anzahl Pixel in der X-Richtung, der Trainingsdaten
const int imageSizeY = ((baseSizeY - 1) * infaltionFactor) + 1 ; //DO NOT CHANGE: Anzahl Pixel in der Y-Richtung, der Trainingsdaten 
const int imagePixels = baseSizeX*baseSizeY; //Anzahl der Pixel eines unskalierten Bildes == Größe eines flachen Vektors eines Bildes
const int num_steps = 1000; //Anzahl an Batches (1000 für die Benchmarks, kann für schnellere Tests reduziert werden)
const int num_trainings_cycles = 3; //Anzahl an Trainingsdurchläufen

//Parameter des CNN
const int num_conv_layers = 1; //Anzahl der Convolutional Layer ()
const int num_filters = 8; //Anzahl der Convolutionen pro Conv-Layer
const int conv_size1 = 3; //DO NOT CHANGE
const int conv_size2 = 3; //DO NOT CHANGE
const int imageSizeX_afterConvolution = imageSizeX - conv_size1 + 1; //DO NOT CHANGE
const int imageSizeY_afterConvolution = imageSizeY - conv_size2 + 1; //DO NOT CHANGE
const int pool_layers_window = 2; //DO NOT CHANGE
const int pool_layers_stride = 2; //DO NOT CHANGE
const int imageSizeX_afterPooling = (imageSizeX_afterConvolution - pool_layers_window) / pool_layers_stride + 1; //DO NOT CHANGE
const int imageSizeY_afterPooling = (imageSizeY_afterConvolution - pool_layers_window) / pool_layers_stride + 1; //DO NOT CHANGE
const int num_weights = 10; //DO NOT CHANGE: Anzahl an Klassifikations Klassen (10, da zehn Ziffern)
const int num_finalImages = pow(num_filters, num_conv_layers); // eigentlich pow(num_filters, num_conv_layers), aber das will nicht
const int num_lastLayer_inputNeurons = num_finalImages * imageSizeX_afterPooling * imageSizeY_afterPooling; //Assert: Durch 100 teilbar

//Parameter der Paralelisierung
const bool parallel = false; //sollen die parallelen Methoden aufgerufen werden?
//const int num_packets = 12; //in wie viele arbeitspakete soll update aufgeteilt werden (falls parallel)

//Konstanten für ADAM, direkt die aus dem Paper
const float alpha = 0.001f; //Lernrate
const float beta1 = 0.9f; //Erstes Moment
const float beta2 = 0.999f; //Zweites Moment
const float EPSILON = 1.0f * powf(10.0f, -8);

#endif /* CONV_H_ */
