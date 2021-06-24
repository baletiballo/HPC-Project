#include "ParallelStuff.h"

JobQueue::JobQueue() {
	abort = false;
}

void JobQueue::push(packaged_task<void()> job) {
	unique_lock<mutex> l(jobQueueMutex);
	jobQueue.push(move(job));
	cv.notify_one();
}

packaged_task<void()> JobQueue::pop() {
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

void JobQueue::terminate() {
	unique_lock<mutex> l(jobQueueMutex);
	abort = true;
	queue<packaged_task<void()>> emptyQueue;
	emptyQueue.swap(jobQueue);
	cv.notify_all();
}

ThreadPool::ThreadPool(int numThreads) {
	threads = numThreads;

	for (int i = 0; i < threads; i++) {
		pool.push_back(thread(&ThreadPool::threadsDoWork, this));
	}
}

void ThreadPool::threadsDoWork() {
	while (true) {
		packaged_task<void()> job = queue.pop();
		if (queue.abort == true) {
			break;
		}
		job();
	}
}

Sem::Sem(int countInit) {
	count = countInit;
}

void Sem::V(int n) {
	unique_lock<mutex> lck(mtx);
	count += n;
	cv.notify_all();
}

void Sem::P(int n) {
	unique_lock<mutex> lck(mtx);
	while (count < n) {
		cv.wait(lck);
	}

	count = -n;
}

void Sem::set(int n) {
	unique_lock<mutex> lck(mtx);
	count = n;
	cv.notify_all();
}

///////////

ThreadPool pool(8);

void endThreads() {
	pool.queue.terminate();
}

void pushJob(packaged_task<void()> job) {
	pool.queue.push(move(job));
}

Sem sem(0);
