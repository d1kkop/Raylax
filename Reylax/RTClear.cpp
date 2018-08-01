#include "ReylaxCuda.h"
using namespace Reylax;

#define CLEAR_THREADS 256

 #define TC 256
 __device__ u32 lk[TC];


// This is faster than writing 4 samples at a time using uint4. 
GLOBAL void rlClearKernel(u32* buffer, u32 numSamples, u32 cv)
{
    u32 addr = bIdx.x * bDim.x + tIdx.x;
    if ( addr >= numSamples ) return;
    buffer[addr] = cv;
  //  atomicInc( lk+(addr%TC), 1UL );
    //atomicAdd2( lk+(addr%TC), 1ULL );
}


extern "C"
void rlClear(u32* buffer, u32 numSamples, u32 clearValue)
{
    dim3 blocks ((numSamples+CLEAR_THREADS-1)/CLEAR_THREADS);
    dim3 threads(CLEAR_THREADS);

#if RL_CUDA
    rlClearKernel<<< blocks, threads >>>
        (
            buffer, numSamples, clearValue
            );
#else
    bDim.x = CLEAR_THREADS;
    for ( u32 b=0; b< blocks.x; b++ )
    {
        bIdx.x = b;
        for ( u32 t=0; t<threads.x; t++ )
        {
            tIdx.x = t;
            rlClearKernel
            (
                buffer, numSamples, clearValue
            );
        }
    }
#endif
}
