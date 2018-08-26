#include "Reylax.h"
#include "Reylax_internal.h"
#include "ReylaxCommon.h"
#include <chrono>
#include <thread>
#include <atomic>
using namespace std;
using namespace chrono;


/*thread_local*/ uint4 cpu_blockDim{};
thread_local uint4 cpu_threadIdx{};
thread_local uint4 cpu_blockIdx{};

namespace Reylax
{
    extern Profiler CpuProfiler;


    void SetDevice(u32 device)
    {
        RL_CUDA_CALL(cudaSetDevice(device));
    }

    u32 GetNumDevices()
    {
        i32 count=0;
        RL_CUDA_CALL(cudaGetDeviceCount(&count));
        return count;
    }

    void SyncDevice()
    {
    #if RL_CUDA
        RL_CUDA_CALL(cudaDeviceSynchronize());
    #endif
    }

    double time()
    {
        return static_cast<double>(duration_cast<duration<double, milli>>(high_resolution_clock::now().time_since_epoch()).count());
    }
    
    void forEachFace(const MeshData** md, u32 numMeshDatas, const std::function<void (u32, const u32[3], const vec3[3])>& cb)
    {
        for (u32 i = 0; i < numMeshDatas ; i++)
        {
            auto* m = md[i];
            const vec3* vd = (const vec3*)m->vertexData[ VERTEX_DATA_POSITION ];
            const u32* ind = m->indices;
            for ( u32 j=0; j<m->numIndices; j+=3 )
            {
                const u32 id[3] = { ind[j], ind[j+1], ind[j+2] };
                const vec3 v[3] = { vd[id[0]], vd[id[1]], vd[id[2]] };
                cb( i, id, v );
            }
        }
    }

    void emulateCpu(const char* profileName, u32 blockDimension, const dim3& blocks, const dim3& threads, const std::function<void ()>& cb)
    {
        if ( blocks.x == 1 && threads.x == 1 ) // Auto detect helper threads
        {
            cb();
            return;
        }
    #if !RL_CUDA
    //    CpuProfiler.start();
        bDim.x = blockDimension;
    #if !RL_CPU_MT
        for ( u32 b=0; b <blocks.x; b++ )
        {
            bIdx.x = b;
            for ( u32 t=0; t<threads.x; t++ )
            {
                tIdx.x = t;
                cb();
            }
        }
    #else
        //static std::thread tds[16];
        //u32 total = blocks.x*threads.x;
        //u32 part = total/16;
        //assert( total % 16 == 0 );
        //for ( u32 i=0; i < 16; ++i )
        //{
        //    tds[i] = std::thread([=]()
        //    {
        //        u32 s = i*part;
        //        u32 e = s+part;
        //        for ( u32 j=s; j<e; ++j )
        //        {
        //            bIdx.x = j/RL_BLOCK_THREADS;
        //            tIdx.x = j-(bIdx.x*RL_BLOCK_THREADS);
        //            cb();
        //        }
        //    });
        //}
        //for ( auto& t : tds ) 
        //    t.join();
        concurrency::parallel_for<u32>(0, blocks.x*threads.x, 1, [&](u32 idx)
        {
            bIdx.x = idx/RL_BLOCK_THREADS;
            tIdx.x = idx-(bIdx.x*RL_BLOCK_THREADS);
            cb();
            /*u32 bPart = blocks.x/4;
            u32 b = idx*bPart;
            u32 e = b+bPart;
            if ( e > blocks.x ) e = blocks.x;
            for ( ; b < e; ++b )
            {*/
            /*              bIdx.x = idx;
                          for ( u32 t=0; t<threads.x; t++ )
                          {
                              tIdx.x = t;
                              cb();
                          }*/
            //}
        });
    #endif
   //     CpuProfiler.stop(profileName);
    #endif
    }

    u32 hostOrDeviceCpy(void* dst, const void* src, u32 size, cudaMemcpyKind kind, bool async)
    {
    #if !RL_CUDA
        if ( dst == memcpy( dst, src, size ) )
            return ERROR_ALL_FINE;
        return ERROR_INVALID_PARAMETER;
    #else
        if ( async )
        {
            RL_CUDA_CALL(cudaMemcpyAsync(dst, src, size, kind));
        }
        else
        {
            RL_CUDA_CALL(cudaMemcpy(dst, src, size, kind));
        }
        return ERROR_ALL_FINE;
    #endif
    }
}