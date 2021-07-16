#ifndef PARAMETER_H_
#define PARAMETER_H_

const int infaltionFactor = 4;//Faktor, um den jedes Bild hochskaliert wird. Also 1 px -> Block mit Kantenlänge infationFactor
const int batchSize = 32; //Anzahl Bilder pro Batch
const int imageSize = 28 * infaltionFactor; //Kantenlänge eines Bildes (nach dem skalieren)
const int imagePixels = 28*28; //Anzahl der Pixel eines unskalierten Bildes == Größe eines flachen Vektors eines Bildes
const int num_steps = 3001; //Anzahl an Batches
const float alpha = 0.001; //Lernrate
const float beta1 = 0.9; //Erstes Moment
const float beta2 = 0.999; //Zweites Moment
const float EPSILON = 1.0f * pow(10.0f, -8);

#endif /* CONV_H_ */