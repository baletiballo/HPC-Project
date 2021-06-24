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

using namespace std;

class JobQueue { //stores packaged Jobs for our threads to complete
public:
	mutex jobQueueMutex;
	queue<packaged_task<void()>> jobQueue;
	condition_variable cv;
	bool abort;

	JobQueue();
	void push(packaged_task<void()> job); //add job to queue
	packaged_task<void()> pop(); //remove first job in queue and returns it
	void terminate(); //empties the queue and makes threads return ending them
};

class ThreadPool { //thread pool of worker threads, getting jobs from the queue
public:
	int threads;
	vector<thread> pool;
	JobQueue queue;

	ThreadPool(int numThreads);
	void threadsDoWork(); //thread method, making them get and complete jobs in an infinite loop
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

void pushJob(packaged_task<void()> job); //push a new job to the queue for the threads to complete

extern Sem sem; //our shared Semaphore for Parallelization

#endif /* PARALLELSTUFF_H_ */
