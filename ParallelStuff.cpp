#include "ParallelStuff.h"

Sem sem(0);
ThreadPool pool(threads);

void endThreads() {
	pool.queue.terminate();
}

void pushJob(int job) {
	pool.queue.push(job);
}

JobQueue::JobQueue() {
	abort = false;
}

void JobQueue::push(int job) {
	std::unique_lock<std::mutex> l(jobQueueMutex);
	jobQueue.push(job);
	cv.notify_one();
}

int JobQueue::pop() {
	std::unique_lock<std::mutex> l(jobQueueMutex);
	cv.wait(l, [this] {
		return abort || !jobQueue.empty();
	});
	if (abort)
		return {};
	int res = jobQueue.front();
	jobQueue.pop();
	return res;
}

void JobQueue::terminate() {
	std::unique_lock<std::mutex> l(jobQueueMutex);
	abort = true;
	std::queue<int> emptyQueue;
	emptyQueue.swap(jobQueue);
	cv.notify_all();
}

ThreadPool::ThreadPool(int numThreads) {
	threads = numThreads;
	currTask = -1;
	c = nullptr;
	m = nullptr;
	f = nullptr;
	cnn = nullptr;

	for (int i = 0; i < threads; i++) {
		pool.push_back(std::thread(&ThreadPool::threadsDoWork, this));
	}
}

void ThreadPool::threadsDoWork() {
	int spot = -1;
	helperMtx.lock();
	spot = spotCounter;
	spotCounter++;
	helperMtx.unlock();
	while (true) {
		int job = queue.pop();
		if (queue.abort == true) {
			return;
		}
		(*cnn).forward(job, spot);
	}
}

void ThreadPool::setTask(int task) {
	currTask = task;
}

void ThreadPool::setConv(Conv &cnew) {
	c = &cnew;
}

void ThreadPool::setMaxPool(MaxPool &mnew) {
	m = &mnew;
}

void ThreadPool::setFullyConnectedLayer(FullyConnectedLayer &fnew) {
	f = &fnew;
}

void ThreadPool::setCNN(CNN &cnnnew) {
	cnn = &cnnnew;
}

Sem::Sem(int countInit) {
	count = countInit;
}

void Sem::V(int n) {
	std::unique_lock<std::mutex> lck(mtx);
	count += n;
	cv.notify_all();
}

void Sem::P(int n) {
	std::unique_lock<std::mutex> lck(mtx);
	while (count < n) {
		cv.wait(lck);
	}

	count = -n;
}

void Sem::set(int n) {
	std::unique_lock<std::mutex> lck(mtx);
	count = n;
	cv.notify_all();
}
