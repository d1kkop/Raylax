#include "Reylax.h"
#include "Reylax_internal.h"

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
    
    void forEachFace(const MeshData* md, u32 numMeshDatas, const std::function<void (u32, const u32[3], const vec3[3])>& cb)
    {
        for (u32 i = 0; i < numMeshDatas ; i++)
        {
            const vec3* vd = (const vec3*)md[i].vertexData[ VERTEX_DATA_POSITION ];
            for ( u32 j=0; j<md[i].numIndices; j+=3 )
            {
                const u32 id[3] = { md[i].indices[j+0], md[i].indices[j+1], md[i].indices[j+2] };
                const vec3 v[3] = { vd[0], vd[1], vd[2] };
                cb( i, id, v );
            }
        }
    }
}