/*
 * ParallelStuff.h
 *
 *  Created on: 24.06.2021
 *      Author: Stefan
 */

#ifndef PARALLELSTUFF_H_
#define PARALLELSTUFF_H_

#pragma once

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>

#include "ReLu.h"
#include "Conv.h"
#include "MaxPool.h"
#include "FullyConnectedLayer.h"

using namespace std;

class JobQueue { //stores packaged Jobs for our threads to complete
public:
	mutex jobQueueMutex;
	queue<int> jobQueue;
	condition_variable cv;
	bool abort;

	JobQueue();
	void push(int job); //add job to queue
	int pop(); //remove first job in queue and returns it
	void terminate(); //empties the queue and makes threads return ending them
};

class ThreadPool { //thread pool of worker threads, getting jobs from the queue
public:
	int threads;
	vector<thread> pool;
	JobQueue queue;
	int currTask;
	Conv *c;
	MaxPool *m;
	FullyConnectedLayer *f;

	ThreadPool(int numThreads);
	void threadsDoWork(); //thread method, making them get and complete jobs in an infinite loop
	void setTask(int task);
	void setConv(Conv &cnew);
	void setMaxPool(MaxPool &mnew);
	void setFullyConnectedLayer(FullyConnectedLayer &fnew);
};

class Sem { //standard semaphore
public:
	mutex mtx;
	condition_variable cv;
	int count;

	Sem(int countInit);

	void V(int n); //add n to count and wakeUp everyone that waits in P()

	void P(int n); //wait until count at least n, then subtract n

	void set(int n); //set count to n, CAUTION: don't do this while someone is waiting in P(), as they won't wake up from this, use V() for that instead
};

///////////

extern ThreadPool pool; //our shared Thread Pool

void endThreads(); //make threads end

void pushJob(int job); //push a new job to the queue for the threads to complete

extern Sem sem; //our shared Semaphore for Parallelization

#endif /* PARALLELSTUFF_H_ */
