#include "Reylax.h"
#include "Reylax_internal.h"
#include <chrono>
using namespace std;
using namespace chrono;

// For now only CUDA implementation. 
// Perhaps OpenCL, later too.

/*thread_local*/ uint4 cpu_blockDim{};
thread_local uint4 cpu_threadIdx{};
thread_local uint4 cpu_blockIdx{};

namespace Reylax
{
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

    void emulateCpu(u32 blockDimension, const dim3& blocks, const dim3& threads, const std::function<void ()>& cb)
    {
    #if !RL_CUDA
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
        concurrency::parallel_for<u32>(0, blocks.x, 1, [&](u32 idx)
        {
            /*for ( u32 b=idx; b <blocks.x; b++ )
            {*/
                bIdx.x = idx;
                for ( u32 t=0; t<threads.x; t++ )
                {
                    tIdx.x = t;
                    cb();
                }
            //}
        });
    #endif
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