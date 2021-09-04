/*
 * ReLu.h
 *
 *  Created on: 24.06.2021
 *      Author: Stefan
 */

#ifndef RELU_H_
#define RELU_H_

#include "parameter.h"
#include "ParallelStuff.h"

void ReLu(float t1 [num_filters] [imageSizeX_afterConvolution] [imageSizeY_afterConvolution]);

void ReLuPrime( float t1 [num_filters] [imageSizeX_afterConvolution] [imageSizeY_afterConvolution],
                float t2  [num_filters] [imageSizeX_afterConvolution] [imageSizeY_afterConvolution]);

/*void ReLu_par();

void ReLuPrime_par();

void ReLuJob(int packet);

void ReLuJobCleanup(int packet);

void ReLuPrimeJob(int packet);

void ReLuJobPrimeCleanup(int packet);*/

#endif /* RELU_H_ */
