/*
 * ReLu.h
 *
 *  Created on: 24.06.2021
 *      Author: Stefan
 */

#ifndef RELU_H_
#define RELU_H_

#pragma once
#include <vector>

using namespace std;


void ReLuJob(int packet);

void ReLuJobCleanup(int packet);

void ReLu(vector<vector<vector<float>>> &t1);

void ReLu_par(vector<vector<vector<float>>> &t1);

void ReLuPrimeJob(int packet);

void ReLuJobPrimeCleanup(int packet);

void ReLuPrime(vector<vector<vector<float>>> &t1, vector<vector<vector<float>>> &t2);

void ReLuPrime_par(vector<vector<vector<float>>> &t1, vector<vector<vector<float>>> &t2);

#endif /* RELU_H_ */
