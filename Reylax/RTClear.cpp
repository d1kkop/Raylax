#include "Reylax_internal.h"
using namespace Reylax;

#define CLEAR_THREADS 256


// This is faster than writing 4 samples at a time using uint4. 
GLOBAL void rlClearKernel(u32* buffer, u32 numSamples, u32 cv)
{
    u32 addr = bIdx.x * bDim.x + tIdx.x;
    if ( addr >= numSamples ) return;
    buffer[addr] = cv;
}


extern "C"
void rlClear(u32* buffer, u32 numSamples, u32 clearValue)
{
    dim3 blocks ((numSamples+CLEAR_THREADS-1)/CLEAR_THREADS);
    dim3 threads(CLEAR_THREADS);
    RL_KERNEL_CALL( CLEAR_THREADS, blocks, threads, rlClearKernel, buffer, numSamples, clearValue );
}
