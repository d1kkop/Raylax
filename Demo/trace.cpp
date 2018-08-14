#include "main.h"
using namespace std;
using namespace glm;
using namespace Reylax;
using namespace chrono;


// Implemented by Reylax


__align__(8)
struct TraceData
{
    vec3 eye;
    mat3 orient;
    vec3* rayDirs;
    u32*  pixels;
};

HOST_OR_DEVICE TraceData TD;
HOST_OR_DEVICE QueueRayFptr QueueRayFunc;


HOST_OR_DEVICE void FirstRays(u32 globalId, u32 localId)
{
    u32 id   = globalId + localId;
    vec3 dir = TD.orient * TD.rayDirs[id];
    vec3 ori = TD.eye;
    QueueRayFunc( localId, &ori.x, &dir.x ); 
}

HOST_OR_DEVICE void TraceCallback(u32 globalId, u32 localId, u32 depth,
                                  const HitResult& hit,
                                  const MeshData* const* meshPtrs, 
                                  const float* ori3, const float* dir3)
{
    TD.pixels[ globalId ] = 200<<8;
}