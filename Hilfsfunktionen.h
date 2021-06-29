/*
 * Hilfsfunktionen.h
 *
 *  Created on: 24.06.2021
 *      Author: Stefan
 */

#ifndef HILFSFUNKTIONEN_H_
#define HILFSFUNKTIONEN_H_

#pragma once
#include <vector>

using namespace std;

vector<float> flatten(vector<vector<vector<float>>> &t1);

vector<vector<vector<float>>> deflatten(vector<float> &t1, int s1, int s2, int s3);

void softmax(vector<float> &t1);


#endif /* HILFSFUNKTIONEN_H_ */
