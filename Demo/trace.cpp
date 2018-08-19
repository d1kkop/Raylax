#include "main.h"
using namespace std;
using namespace chrono;



HOST_OR_DEVICE TraceData TD;
HOST_OR_DEVICE QueueRayFptr QueueRayFunc;


template <typename T>
void SetSymbol(T& dst, const void* src, bool wait=false)
{
#if !DEMO_CPU
    cudaError err;
    if ( !wait ) err = cudaMemcpyToSymbolAsync(dst, src, sizeof(T), 0, cudaMemcpyDefault);
    else         err = cudaMemcpyToSymbol(dst, src, sizeof(T), 0, cudaMemcpyDefault);
    assert(err==0);
#else
    memcpy(&dst, src, sizeof(T));
#endif
}


void UpdateTraceData(const TraceData& td, QueueRayFptr queueRayFptr)
{
    static bool firstTime=true;
    if ( firstTime )
    {
        firstTime=false;
        SetSymbol( QueueRayFunc, &queueRayFptr );
    }
    SetSymbol( TD, &td );
}

HOST_OR_DEVICE vec3 interpolate3(const HitResult& hit, const MeshData* meshData, u32 dataIdx)
{
    assert(meshData);
    assert(dataIdx < VERTEX_DATA_COUNT);
    const MeshData* mesh = &meshData[hit.face->w];
    const vec3* vd  = (const vec3*) mesh->vertexData[dataIdx];
    const vec3& vd1 = vd[hit.face->x];
    const vec3& vd2 = vd[hit.face->y];
    const vec3& vd3 = vd[hit.face->z];
    float u = hit.u;
    float v = hit.v;
    float w = 1-(u+v);
    return vd1*u + vd2*v + vd3*w;
}

HOST_OR_DEVICE void FirstRays(u32 globalId, u32 localId)
{
    vec3 dir = TD.orient * TD.rayDirs[globalId];
    vec3 ori = TD.eye;
    QueueRayFunc( &ori.x, &dir.x ); 
}

HOST_OR_DEVICE void TraceCallback(u32 globalId, u32 localId, u32 depth,
                                  const HitResult& hit,
                                  const MeshData* const* meshPtrs)
{
 //   auto mesh = meshPtrs[ hit.face->w ];
  //  vec3 n = interpolate3( hit, mesh, VERTEX_DATA_NORMAL );
//    vec3 refl = reflect( hit.rd, n );
    TD.pixels[ globalId ] = 200<<8;
}