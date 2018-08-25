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


template <typename T>
void GetSymbol(void* dst, const T& src, bool wait=true)
{
#if !DEMO_CPU
    cudaError err;
    if ( !wait ) err = cudaMemcpyFromSymbolAsync(dst, src, sizeof(T), 0, cudaMemcpyDeviceToHost);
    else         err = cudaMemcpyFromSymbol(dst, src, sizeof(T), 0, cudaMemcpyDeviceToHost);
    assert(err==0);
#else
    memcpy(dst, &src, sizeof(T));
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

template <class T>
HOST_OR_DEVICE T Interpolate(const HitResult& hit, const MeshData* const* meshPtrs, u32 dataIdx)
{
    assert(meshPtrs);
    assert(dataIdx < VERTEX_DATA_COUNT);
    const MeshData* mesh = meshPtrs[hit.face->w];
    const T* vd  = (const T*) (mesh->vertexData[dataIdx]);
    const T& vd1 = vd[hit.face->x];
    const T& vd2 = vd[hit.face->y];
    const T& vd3 = vd[hit.face->z];
    float u = hit.u;
    float v = hit.v;
    float w = 1-(u+v);
    return w*vd1 + u*vd2 + v*vd3;
}

HOST_OR_DEVICE u32 rgba(const vec4& c)
{
    u32 r = (u32)(c.x*255.f);
    u32 g = (u32)(c.y*255.f);
    u32 b = (u32)(c.z*255.f);
    u32 a = (u32)(c.w*255.f);
    if ( r > 255 ) r = 255;
    if ( g > 255 ) g = 255;
    if ( b > 255 ) b = 255;
    if ( a > 255 ) a = 255;
    return (a<<24)|(r<<16)|(g<<8)|(b);
}

HOST_OR_DEVICE u32 single(float f)
{
    u32 r = (u32)(f*255.f);
    if ( r > 255 ) r = 255;
    return r;
}

HOST_OR_DEVICE void FirstRays(u32 globalId, u32 localId)
{
    printf("yeeey %d %d\n", globalId, localId);
    vec3 dir = /*TD.orient **/ TD.rayDirs[globalId];
    vec3 ori = /*TD.eye;*/ vec3(0,0,-2.5f);
  //  QueueRayFunc( &ori.x, &dir.x ); 
}

HOST_OR_DEVICE void TraceCallback(u32 globalId, u32 localId, u32 depth,
                                  const HitResult& hit,
                                  const MeshData* const* meshPtrs)
{
    vec3 n = Interpolate<vec3>(hit, meshPtrs, VERTEX_DATA_NORMAL);
    n = normalize(n);
    TD.pixels[globalId] = single(abs(n.z)) << 16;
}


HOST_OR_DEVICE RaySetupFptr firstRaysFptr = FirstRays;
RaySetupFptr GetSetupFptr()
{
    RaySetupFptr res;
    GetSymbol(&res, firstRaysFptr);
    return res;
}

HOST_OR_DEVICE HitResultFptr hitResultFptr = TraceCallback;
HitResultFptr GetHitFptr()
{
    HitResultFptr res;
    GetSymbol(&res, hitResultFptr);
    return res;
}
