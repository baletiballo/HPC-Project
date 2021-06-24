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
	bool abort = false;

	void push(packaged_task<void()> job) {
		unique_lock<mutex> l(jobQueueMutex);
		jobQueue.push(move(job));
		cv.notify_one();
	}

	packaged_task<void()> pop() {
		unique_lock<mutex> l(jobQueueMutex);
		cv.wait(l, [this] {
			return abort || !jobQueue.empty();
		});
		if (abort)
			return {};
		auto r = move(jobQueue.front());
		jobQueue.pop();
		return move(r);
	}

	void terminate() {
		unique_lock<mutex> l(jobQueueMutex);
		abort = true;
		queue<packaged_task<void()>> emptyQueue;
		emptyQueue.swap(jobQueue);
		cv.notify_all();
	}
};

class ThreadPool {
public:
	int threads;
	vector<thread> pool;
	JobQueue queue;

	ThreadPool(int numThreads) {
		threads = numThreads;

		for (int i = 0; i < threads; i++) {
			pool.push_back(thread(&threadsDoWork, this));
		}
	}

	void threadsDoWork() {
		while (true) {
			packaged_task<void()> job = queue.pop();
			if (queue.abort == true) {
				break;
			}
			job();
		}
	}
};

class Sem {
public:
	mutex mtx;
	condition_variable cv;
	int count;

	Sem(int countInit) {
		count = countInit;
	}

	void V(int n) {
		unique_lock<mutex> lck(mtx);
		count += n;
		cv.notify_all();
	}

	void P(int n) {
		unique_lock<mutex> lck(mtx);
		while (count < n) {
			cv.wait(lck);
		}

		count = -n;
	}

	void set(int n) {
		unique_lock<mutex> lck(mtx);
		count = n;
		cv.notify_all();
	}
};

///////////

ThreadPool pool(8);

void endThreads() {
	pool.queue.terminate();
}

void pushJob(packaged_task<void()> job) {
	pool.queue.push(move(job));
}

Sem sem(0);