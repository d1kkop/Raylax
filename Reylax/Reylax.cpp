#include "Reylax.h"
#include "Reylax_internal.h"
#include <chrono>
using namespace std;
using namespace chrono;

// For now only CUDA implementation. 
// Perhaps OpenCL, later too.

namespace Reylax
{
    void setDevice(u32 device)
    {
        RL_CUDA_CALL(cudaSetDevice(device));
    }

    u32 getNumDevices()
    {
        i32 count=0;
        RL_CUDA_CALL(cudaGetDeviceCount(&count));
        return count;
    }

    void syncDevice()
    {
        RL_CUDA_CALL(cudaDeviceSynchronize());
    }

    double time()
    {
        return static_cast<double>(duration_cast<duration<double, milli>>(high_resolution_clock::now().time_since_epoch()).count());
    }
    
    void forEachFace(const MeshData** md, u32 numMeshDatas, const std::function<void (u32, const u32[3], const vec3[3])>& cb)
    {
        for (u32 i = 0; i < numMeshDatas ; i++)
        {
            const vec3* vd = (const vec3*)md[i]->vertexData[ VERTEX_DATA_POSITION ];
            for ( u32 j=0; j<md[i]->numIndices; j+=3 )
            {
                const u32 id[3] = { md[i]->indices[j+0], md[i]->indices[j+1], md[i]->indices[j+2] };
                const vec3 v[3] = { vd[id[0]], vd[id[1]], vd[id[2]] };
                cb( i, id, v );
            }
        }
    }

    void emulateCpu(u32 blockDim, const dim3& blocks, const dim3& threads, const std::function<void ()>& cb)
    {
    #if !RL_CUDA
        bDim.x = blockDim;
        for ( u32 b=0; b <blocks.x; b++ )
        {
            bIdx.x = b;
            for ( u32 t=0; t<threads.x; t++ )
            {
                tIdx.x = t;
                cb();
            }
        }
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