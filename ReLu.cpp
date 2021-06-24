#include "ReLu.h"
#include "ParallelStuff.h"

int reluPackets = 64;
int reluPacketSize=0;
int reluSize1=0;
int reluSize2=0;
int reluSize3=0;
vector<vector<vector<float>>> *relu_input = nullptr;
vector<vector<vector<float>>> *relu_input_2 = nullptr;

void ReLuJob(int packet) {
	for (int i = packet * reluPacketSize; i < (packet + 1) * reluPacketSize; i++) {
		int index1 = i / (reluSize2 * reluSize3);
		int index2 = (i / reluSize3) % reluSize2;
		int index3 = i % reluSize3;
		if ((*relu_input)[index1][index2][index3] <= 0) {
			(*relu_input)[index1][index2][index3] = 0;
		}
	}
	sem.V(1);
}

void ReLuJobCleanup(int packet) {
	for (int i = packet * reluPacketSize; i < (reluSize1 * reluSize2 * reluSize3); i++) {
		int index1 = i / (reluSize2 * reluSize3);
		int index2 = (i / reluSize3) % reluSize2;
		int index3 = i % reluSize3;
		if ((*relu_input)[index1][index2][index3] <= 0) {
			(*relu_input)[index1][index2][index3] = 0;
		}
	}
}

void ReLu(vector<vector<vector<float>>> &t1) {
	reluSize1 = t1.size();
	reluSize2 = t1[0].size();
	reluSize3 = t1[0][0].size();
	reluPacketSize = (reluSize1 * reluSize2 * reluSize3) / reluPackets;
	relu_input = &t1;

	sem.set(0);
	for (int i = 0; i < reluPackets; i++) {
		packaged_task<void()> job(bind(&ReLuJob, i));
		pushJob(move(job));
	}
	if ((reluSize1 * reluSize2 * reluSize3) % reluPackets != 0) {
		ReLuJobCleanup(reluPackets + 1);
	}
	sem.P(reluPackets);
}

void ReLuPrimeJob(int packet) {
	for (int i = packet * reluPacketSize; i < (packet + 1) * reluPacketSize; i++) {
		int index1 = i / (reluSize2 * reluSize3);
		int index2 = (i / reluSize3) % reluSize2;
		int index3 = i % reluSize3;
		if ((*relu_input_2)[index1][index2][index3] <= 0) {
			(*relu_input)[index1][index2][index3] = 0;
		}
	}
	sem.V(1);
}

void ReLuJobPrimeCleanup(int packet) {
	for (int i = packet * reluPacketSize; i < (reluSize1 * reluSize2 * reluSize3); i++) {
		int index1 = i / (reluSize2 * reluSize3);
		int index2 = (i / reluSize3) % reluSize2;
		int index3 = i % reluSize3;
		if ((*relu_input_2)[index1][index2][index3] <= 0) {
			(*relu_input)[index1][index2][index3] = 0;
		}
	}
}

void ReLuPrime(vector<vector<vector<float>>> &t1, vector<vector<vector<float>>> &t2) {
	reluSize1 = t1.size();
	reluSize2 = t1[0].size();
	reluSize3 = t1[0][0].size();
	reluPacketSize = (reluSize1 * reluSize2 * reluSize3) / reluPackets;
	relu_input = &t1;
	relu_input_2 = &t2;

	sem.set(0);
	for (int i = 0; i < reluPackets; i++) {
		packaged_task<void()> job(bind(&ReLuPrimeJob, i));
		pushJob(move(job));
	}
	if ((reluSize1 * reluSize2 * reluSize3) % reluPackets != 0) {
		ReLuJobPrimeCleanup(reluPackets + 1);
	}
	sem.P(reluPackets);
}

/*void ReLu(vector<vector<vector<float>>> &t1) {
	for (unsigned i = 0; i < t1.size(); i++) {
		for (unsigned j = 0; j < t1[i].size(); j++) {
			for (unsigned k = 0; k < t1[i][j].size(); k++) {
				if (t1[i][j][k] <= 0) {
					t1[i][j][k] = 0;
				}
			}
		}
	}
}

void ReLuPrime(vector<vector<vector<float>>> &t1, vector<vector<vector<float>>> &t2) {
	for (unsigned i = 0; i < t1.size(); i++) {
		for (unsigned j = 0; j < t1[i].size(); j++) {
			for (unsigned k = 0; k < t1[i][j].size(); k++) {
				if (t2[i][j][k] <= 0) {
					t1[i][j][k] = 0;
				}
			}
		}
	}
}*/
