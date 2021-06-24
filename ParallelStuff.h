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

class JobQueue {
public:
	mutex jobQueueMutex;
	queue<packaged_task<void()>> jobQueue;
	condition_variable cv;
	bool abort;

	JobQueue();
	void push(packaged_task<void()> job);
	packaged_task<void()> pop();
	void terminate();
};

class ThreadPool {
public:
	int threads;
	vector<thread> pool;
	JobQueue queue;

	ThreadPool(int numThreads);
	void threadsDoWork();
};

class Sem {
public:
	mutex mtx;
	condition_variable cv;
	int count;

	Sem(int countInit);

	void V(int n);

	void P(int n);

	void set(int n);
};

///////////

extern ThreadPool pool;

void endThreads();

void pushJob(packaged_task<void()> job);

extern Sem sem;

#endif /* PARALLELSTUFF_H_ */
