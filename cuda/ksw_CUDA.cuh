#include "stdint.h"
#include "batch_config.h"
#include <math.h>

#define KSW_XBYTE  0x10000
#define KSW_XSTOP  0x20000
#define KSW_XSUBO  0x40000
#define KSW_XSTART 0x80000
#define KSW_MAX_QLEN SEQ_MAXLEN
#define WARPSIZE 32

#ifndef KSW_MAX2
#define KSW_MAX2(a,b) ((a)>(b)?(a):(b))
#endif
#define KSW_ALL_THREADS 0xffffffff  // mask indicating all threads participate in shuffle instruction

typedef	struct m128i {
	// set of 8 16-bit integers
	int16_t x0, x1, x2, x3, x4, x5, x6, x7;
} m128i;

typedef struct kswq_t {
	m128i *qp, *H0, *H1, *E, *Hmax;
	int qlen, slen;
	uint8_t shift, mdiff, max, size;
} kswq_t;

typedef struct {
	int score; // best score
	int te, qe; // target end and query end
	int score2, te2; // second best score and ending position on the target
	int tb, qb; // target start and query start
} kswr_t;

__device__ kswr_t ksw_align2(int qlen, uint8_t *query, int tlen, uint8_t *target, int m, const int8_t *mat, int o_del, int e_del, int o_ins, int e_ins, int xtra, kswq_t **qry, void* d_buffer_ptr);

__device__ int ksw_extend2(int qlen, const uint8_t *query, int tlen, const uint8_t *target, int m, const int8_t *mat, int o_del, int e_del, int o_ins, int e_ins, int w, int end_bonus, int zdrop, int h0, int *_qle, int *_tle, int *_gtle, int *_gscore, int *_max_off, void* d_buffer_ptr);

/* SW extension function to be executed at warp level. bandwidth not applied */
__device__ int ksw_extend_warp(int qlen, const uint8_t *query, int tlen, const uint8_t *target, int m, const int8_t *mat, int o_del, int e_del, int o_ins, int e_ins, int h0, int *_qle, int *_tle, int *_gtle, int *_gscore);
/* SW global function to be executed at warp level. bandwidth not applied */
__device__ int ksw_global_warp(int qlen, const uint8_t *query, int tlen, const uint8_t *target, int m, const int8_t *mat, int o_del, int e_del, int o_ins, int e_ins, int *i_max, int *j_max, uint8_t *traceback);

__device__ int ksw_global2(int qlen, const uint8_t *query, int tlen, const uint8_t *target, int m, const int8_t *mat, int o_del, int e_del, int o_ins, int e_ins, int w, int *n_cigar_, uint32_t **cigar_, void* d_buffer_ptr);

/* scoring of 2 characters given scoring matrix mat, and dimension m */
static __device__ __forceinline__ int ksw_score(uint8_t A, uint8_t B, const int8_t *mat, int m){
	return (int)mat[A*m+B];
}

/* SW extension executing at warp level
	BLOCKSIZE = WARPSIZE = 32
	requires at least qlen*4 bytes of shared memory
	currently implemented at 500*4 bytes of shared mem	
	return max score in the matrix, qle, tle, gtle, gscore
	NOTATIONS:
		SM_H[], SM_E: shared memory arrays for storing H and E of thread 31 for transitioning between tiles
		e, f, h     : E[i,j], F[i,j], H[i,j] to be calculated in an iteration
		e1_			: E[i-1,j] during a cell calculation
		h1_,h_1,h11 : // H[i-1,j], H[i,j-1], H[i-1,j-1]
		max_score   : the max score that a thread has found
		i_m, j_m	: the position where we found max_score
	CALCULATION:
		E[i,j] = max(H[i-1,j]-gap_open_penalty, E[i-1,j]-gap_ext_penalty)
		F[i,j] = max(H[i,j-1]-gap_open_penalty, F[i,j-1]-gap_ext_penalty)
		H[i,j] = max(0, E[i,j], F[i,j], H[i-1.j-1]+score(query[j],target[i]))
 */
static __device__ __forceinline__ int ksw_extend_warp(int qlen, const uint8_t *query, int tlen, const uint8_t *target, int m, const int8_t *mat, int o_del, int e_del, int o_ins, int e_ins, int h0, int *_qle, int *_tle, int *_gtle, int *_gscore)
{
	if (qlen>KSW_MAX_QLEN){printf("querry length is too long %d \n", qlen); __trap();}
	__shared__	int16_t SM_H[KSW_MAX_QLEN], SM_E[KSW_MAX_QLEN];
	int e, f, h;
	int e1_;
	int h1_, h_1, h11;
	int max_score = h0;	// best score
	int i_m=-1, j_m=-1;	// position of best score
	int max_gscore = 0; // score of end-to-end alignment
	int i_gscore;	// position of best end-to-end alignment score

	// first row scoring
	for (int j=threadIdx.x; j<qlen; j+=WARPSIZE){	// j is col index
		SM_E[j] = 0;
		h = h0 - o_ins - e_ins - j*e_ins;
		SM_H[j] = (h>0)? h : 0;
	}
	// first we fill the top-left corner where we don't have enough parallelism
	f = 0;	// first column of F
	for (int anti_diag=0; anti_diag<WARPSIZE-1; anti_diag++){
		int i = threadIdx.x; 				// row index on the matrix
		int j = anti_diag - threadIdx.x;	// col index on the matrix
		// get previous cell data
		e1_ = __shfl_up_sync(KSW_ALL_THREADS, e, 1); // get e from threadIdx-1, which is E[i-1,j]
		if (threadIdx.x==0) e1_ = 0;
		h1_ = __shfl_up_sync(KSW_ALL_THREADS, h, 1); // h from threadID-1 is H[i-1,j]
		if (threadIdx.x==0) h1_ = SM_H[j];	   // but row 0 get initial scoring from shared mem
		h11 = __shfl_up_sync(KSW_ALL_THREADS, h_1, 1); // h_1 from threadID-1 is H[i-1,j-1]
		if (threadIdx.x==0 && j!=0) h11 = SM_H[j-1];	// row 0 get initial scoring from shared mem, except for first column
		if (threadIdx.x==0 && j==0) h11 = h0;			// H[-1,-1] = h0
		h_1 = h;							// H[i,j-1] from previous iteration of same thread 
		if (j==0) h_1 = h0 - o_ins - (i+1)*e_ins;		// first column score
		// calculate E[i,j], F[i,j], and H[i,j]
		if (i<tlen && j<qlen && j>=0){ 		// safety check for small matrix
			e = KSW_MAX2(h1_-o_del-e_del, e1_-e_del);
			f = KSW_MAX2(h_1-o_ins-e_ins, f-e_ins);
			h = h11 + ksw_score(target[i], query[j], mat, m);
			h = KSW_MAX2(0, h);
			int tmp = KSW_MAX2(e,f);
			h = KSW_MAX2(tmp, h);
			// record max scoring
			if (h>max_score){
				max_score = h; i_m = i; j_m = j;
			}
			if (j==qlen-1){	// we have hit last column
				if (h>max_gscore)	// record max to-end alignment score
					{max_gscore = h; i_gscore = i;}
			}
		}
	}

	// fill the rest of the matrix where we have enough parallelism
	int Ntile = ceil((float)tlen/WARPSIZE);
	int qlen_padded = qlen>=32? qlen : 32;	// pad qlen so that we have correct overflow for small matrix
	for (int tile_ID=0; tile_ID<Ntile; tile_ID++){	// tile loop
		int i, j;
		for (int anti_diag=WARPSIZE-1; anti_diag<qlen_padded+WARPSIZE-1; anti_diag++){	// anti-diagonal loop
			i = tile_ID*WARPSIZE + threadIdx.x;	// row index on matrix
			j = anti_diag - threadIdx.x; 		// col index
			if (j>=qlen_padded){			// when hit the end of this tile, overflow to next tile
				i = i+WARPSIZE;		// over flow to its row on the next tile
				j = j-qlen_padded;			// reset col index to the first 31 columns on next tile
			}
			// __syncwarp();
			// get previous cell data
			if (j==0) f = 0; 	// if we are processing first col, F[i,j-1] = 0. Otherwise, F[i,j-1] = f
			e1_ = __shfl_up_sync(KSW_ALL_THREADS, e, 1); 	// get e from threadIdx-1, which is E[i-1,j]
			if (threadIdx.x==0) e1_ = SM_E[j];	// thread 0 get E[i-1] from shared mem, which came from thread 31 of previous tile
			h1_ = __shfl_up_sync(KSW_ALL_THREADS, h, 1); 	// h from threadID-1 is H[i-1,j]
			if (threadIdx.x==0) h1_ = SM_H[j];	// but row 0 get initial scoring from shared mem, which came from thread 31 of previous tile
			h11 = __shfl_up_sync(KSW_ALL_THREADS, h_1, 1); // h_1 from threadID-1 is H[i-1,j-1]
			if (threadIdx.x==0 && j!=0) h11 = SM_H[j-1];	// thread 0 get H[i-1,j-1] from shared mem, which came from thread 31
			if (threadIdx.x==0 && j==0) h11 = h0 - o_ins - i*e_ins;	// first column scoring
			h_1 = h;							// H[i,j-1] from previous iteration of same thread 
			if (j==0) h_1 = h0 - o_ins - (i+1)*e_ins;	// first column score
			// calculate E[i,j], F[i,j], and H[i,j]
			if (i<tlen && j<qlen){ // j should be >=0
				e = KSW_MAX2(h1_-o_del-e_del, e1_-e_del);
				f = KSW_MAX2(h_1-o_ins-e_ins, f-e_ins);
				h = h11 + ksw_score(target[i], query[j], mat, m);
				h = KSW_MAX2(0, h);
				int tmp = KSW_MAX2(e,f);
				h = KSW_MAX2(tmp, h);
				// record max scoring
				if (h>max_score){
					max_score = h; i_m = i; j_m = j;
				}
				// thread 31 need to write h and e to shared memory to serve thread 0 in the next tile
				if (threadIdx.x==31){ SM_H[j] = h; SM_E[j] = e; }
				if (j==qlen-1){	// we have hit last column
					if (h>max_gscore)	// record max to-end alignment score
						{max_gscore = h; i_gscore = i;}
				}
			}
		}
	}
	// finished filling the matrix, now we find the max of max_score across the warp
	// use reduction to find the max of 32 max's
	for (int i=0; i<5; i++){
		int tmp = __shfl_down_sync(KSW_ALL_THREADS, max_score, 1<<i);
		int tmp_i = __shfl_down_sync(KSW_ALL_THREADS, i_m, 1<<i);
		int tmp_j = __shfl_down_sync(KSW_ALL_THREADS, j_m, 1<<i);
		if (max_score < tmp) {max_score = tmp; i_m = tmp_i; j_m = tmp_j;}
		tmp = __shfl_down_sync(KSW_ALL_THREADS, max_gscore, 1<<i);
		tmp_i = __shfl_down_sync(KSW_ALL_THREADS, i_gscore, 1<<i);
		if (max_gscore < tmp){max_gscore = tmp; i_gscore = tmp_i;}
	}

	// write max, i_m, j_m to global memory
	if (_qle) *_qle = j_m + 1;
	if (_tle) *_tle = i_m + 1;
	if (_gtle) *_gtle = i_gscore + 1;
	if (_gscore) *_gscore = max_gscore;
	return max_score;	// only thread 0's result is valid
}

