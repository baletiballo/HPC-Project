#include "ReLu.h"

/*int reluPackets = num_packets;
int reluPacketSize = 0;*/
const int reluSize1 = num_filters;
const int reluSize2 = imageSizeX_afterConvolution;
const int reluSize3 = imageSizeY_afterConvolution;
/*float (*relu_input) [reluSize2] [reluSize3];
float (*relu_input_2) [reluSize2] [reluSize3];*/

void ReLu(float t1 [reluSize1] [reluSize2] [reluSize3]) {
	/*if (parallel)
	{
		relu_input = t1;
		ReLu_par();
		return;
	}
	*/
	for (unsigned i = 0; i < reluSize1; i++) {
		for (unsigned j = 0; j < reluSize2; j++) {
			for(unsigned k = 0; k < reluSize3; k+=10){
				//DEPENDS ON: reluSize3
				//if (t1[i][j][k] <= 0) 
				//t1[i][j][k] = 0;
				if (t1[i][j][k+0] <= 0)
					t1[i][j][k+0] = 0;

				if (t1[i][j][k+1] <= 0)
					t1[i][j][k+1] = 0;

				if (t1[i][j][k+2] <= 0)
					t1[i][j][k+2] = 0;

				if (t1[i][j][k+3] <= 0)
					t1[i][j][k+3] = 0;

				if (t1[i][j][k+4] <= 0)
					t1[i][j][k+4] = 0;

				if (t1[i][j][k+5] <= 0)
					t1[i][j][k+5] = 0;

				if (t1[i][j][k+6] <= 0)
					t1[i][j][k+6] = 0;

				if (t1[i][j][k+7] <= 0)
					t1[i][j][k+7] = 0;

				if (t1[i][j][k+8] <= 0)
					t1[i][j][k+8] = 0;

				if (t1[i][j][k+9] <= 0)
					t1[i][j][k+9] = 0;
			}
		}
	}
}

void ReLuPrime(float t1 [reluSize1] [reluSize2] [reluSize3], float t2  [reluSize1] [reluSize2] [reluSize3]) {
	/*if (parallel)
	{
		relu_input = t1;
		relu_input_2 = t2;
		ReLuPrime_par();
		return;
	} 
	*/
	for (unsigned i = 0; i < reluSize1; i++) {
		for (unsigned j = 0; j < reluSize2; j++) {
			for(unsigned k = 0; k < reluSize3; k+=10){
				//DEPENDS ON: reluSize3
				//if (t2[i][j][k] <= 0) 
				//t1[i][j][k] = 0;
				if (t2[i][j][0+k] <= 0)
					t1[i][j][0+k] = 0;

				if (t2[i][j][1+k] <= 0)
					t1[i][j][1+k] = 0;

				if (t2[i][j][2+k] <= 0)
					t1[i][j][2+k] = 0;

				if (t2[i][j][3+k] <= 0)
					t1[i][j][3+k] = 0;

				if (t2[i][j][4+k] <= 0)
					t1[i][j][4+k] = 0;

				if (t2[i][j][5+k] <= 0)
					t1[i][j][5+k] = 0;

				if (t2[i][j][6+k] <= 0)
					t1[i][j][6+k] = 0;

				if (t2[i][j][7+k] <= 0)
					t1[i][j][7+k] = 0;

				if (t2[i][j][8+k] <= 0)
					t1[i][j][8+k] = 0;

				if (t2[i][j][9+k] <= 0)
					t1[i][j][9+k] = 0;
			}
		}
	}
}

//void ReLuJob(int packet) {
//	for (int i = packet * reluPacketSize; i < (packet + 1) * reluPacketSize; i++) {
//		int index1 = i / (reluSize2 * reluSize3);
//		int index2 = (i / reluSize3) % reluSize2;
//		int index3 = i % reluSize3;
//
//		if (relu_input[index1][index2][index3] <= 0)
//			relu_input[index1][index2][index3] = 0;
//	}
//
//	sem.V(1);
//}
//
//void ReLuJobCleanup(int packet) {
//	for (int i = packet * reluPacketSize; i < (reluSize1 * reluSize2 * reluSize3); i++) {
//		int index1 = i / (reluSize2 * reluSize3);
//		int index2 = (i / reluSize3) % reluSize2;
//		int index3 = i % reluSize3;
//
//		if (relu_input[index1][index2][index3] <= 0)
//			relu_input[index1][index2][index3] = 0;
//	}
//}
//
//void ReLu_par() {
//	reluPacketSize = (reluSize1 * reluSize2 * reluSize3) / reluPackets;
//
//	sem.set(0);
//	pool.setTask(7);
//	for (int i = 0; i < reluPackets; i++)
//		pushJob(i);
//
//	if ((reluSize1 * reluSize2 * reluSize3) % reluPackets != 0)
//		ReLuJobCleanup(reluPackets + 1);
//
//	sem.P(reluPackets);
//}
//
//void ReLuPrimeJob(int packet) {
//	for (int i = packet * reluPacketSize; i < (packet + 1) * reluPacketSize; i++) {
//		int index1 = i / (reluSize2 * reluSize3);
//		int index2 = (i / reluSize3) % reluSize2;
//		int index3 = i % reluSize3;
//
//		if (relu_input_2[index1][index2][index3] <= 0)
//			relu_input[index1][index2][index3] = 0;
//	}
//
//	sem.V(1);
//}
//
//void ReLuJobPrimeCleanup(int packet) {
//	for (int i = packet * reluPacketSize; i < (reluSize1 * reluSize2 * reluSize3); i++) {
//		int index1 = i / (reluSize2 * reluSize3);
//		int index2 = (i / reluSize3) % reluSize2;
//		int index3 = i % reluSize3;
//
//		if (relu_input_2[index1][index2][index3] <= 0)
//			relu_input[index1][index2][index3] = 0;
//	}
//}
//
//void ReLuPrime_par() {
//	reluPacketSize = (reluSize1 * reluSize2 * reluSize3) / reluPackets;
//
//	sem.set(0);
//	pool.setTask(8);
//	for (int i = 0; i < reluPackets; i++)
//		pushJob(i);
//
//	if ((reluSize1 * reluSize2 * reluSize3) % reluPackets != 0)
//		ReLuJobPrimeCleanup(reluPackets + 1);
//
//	sem.P(reluPackets);
//}


