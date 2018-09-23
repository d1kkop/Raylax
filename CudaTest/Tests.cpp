#include "UnitTestBase.h"
#include <thread>
#include <mutex>
#include <atomic>
#include <ppl.h>
#include <cuda_runtime.h>
using namespace std;
using namespace concurrency;

UTESTBEGIN(PreheadThreads)
{
    int num=10000000;
     int p = 0;
   // atomic_int p = 0;
    parallel_for(0, num, 1, [&](int i)
    {
        p += 1;
    });
    int res = p;
    printf("Result = %d\n", res);
    return true;
}
UNITTESTEND(PreheadThreads)


UTESTBEGIN(MtSpeed)
{
    int num=10000000;
    int p = 0;
    //atomic_int p =0;
    std::thread threads[4];
    for ( auto& t : threads )
    {
        t = std::thread([&]()
        {
            for ( int i=0; i<num/4; ++i )
            {
                p += 1;
            }
        });
    }
    for ( auto& t : threads ) t.join();
    int res = p;
    printf("Result = %d\n", res);
    return true;
}
UNITTESTEND(MtSpeed)

UTESTBEGIN(PreheadThreads2)
{
    int num=10000000;
    int p = 0;
   // atomic_int p = 0;
    parallel_for(0, num, 1, [&](int i)
    {
        p += 1;
    });
    int res = p;
    printf("Result = %d\n", res);
    return true;
}
UNITTESTEND(PreheadThreads2)


UTESTBEGIN(SingleThreadedT)
{
    int num=10000000;
    // int p = 0;
    atomic_int p = 0;
    for ( int i=0; i<num; ++i )
    {
        p += 1;
    }
    int res = p;
    printf("Result = %d\n", res);
    return true;
}
UNITTESTEND(SingleThreadedT)

UTESTBEGIN(SingleThreadedNonAtomic)
{
    int num=10000000;
    int p = 0;
    // atomic_int p = 0;
    #pragma omp parallel num_threads(4)
    {
        for ( int i=0; i<num>>2; ++i )
        {
            p += 1;
        }
    }
    int res = p;
    printf("Result = %d\n", res);
    return true;
}
UNITTESTEND(SingleThreadedNonAtomic)

UTESTBEGIN(SingleThreadedVolatile)
{
    int num=10000000;
    //volatile int p = 0;
    atomic_int p = 0;
    #pragma omp parallel num_threads(4)
    {
        for ( int i=0; i<num>>2; ++i )
        {
            p += 1;
        }
    }
    int res = p;
    printf("Result = %d\n", res);
    return true;
}
UNITTESTEND(SingleThreadedVolatile)


__device__ int d_p = 0;
__global__ void add_p()
{
    //atomicAdd( &d_p, 1 );
    d_p += 1;
}

UTESTBEGIN(CudaTestAdd1)
{
    int num=10000000;
    dim3 blocks  (( num+255)/256 );
    dim3 threads (256);
    for ( int i = 0; i <10; i++ )
    {
        add_p<<<blocks, threads>>>();
    }
    return true;
}
UNITTESTEND(CudaTestAdd1)