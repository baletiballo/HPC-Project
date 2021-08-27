/*
 * ReLu.h
 *
 *  Created on: 24.06.2021
 *      Author: Stefan
 */

#ifndef RELU_H_
#define RELU_H_

#include "parameter.h"
#include "Conv.h"
#include "ParallelStuff.h"


int reluPackets = num_packets;
int reluPacketSize = 0;
const int reluSize1 = num_filters;
const int reluSize2 = imageSizeX_afterConvolution;
const int reluSize3 = imageSizeX_afterConvolution;
float (*relu_input) [reluSize2] [reluSize3];
float (*relu_input_2) [reluSize2] [reluSize3];

void ReLu(float t1 [reluSize1] [reluSize2] [reluSize3]);

void ReLuPrime(float t1 [reluSize1] [reluSize2] [reluSize3], float t2  [reluSize1] [reluSize2] [reluSize3]);

void ReLu_par();

void ReLuPrime_par();

void ReLuJob(int packet);

void ReLuJobCleanup(int packet);

void ReLuPrimeJob(int packet);

void ReLuJobPrimeCleanup(int packet);

#endif /* RELU_H_ */
