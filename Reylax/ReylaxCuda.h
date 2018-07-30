#pragma once
#include "Cuda.h"
#include "glm/common.hpp"
#include "glm/geometric.hpp"
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/transform.hpp"
#include "glm/gtx/rotate_vector.hpp"
#include "ReylaxTypes.h"
using namespace glm;


#if RL_CUDA 

#define GLOBAL __global__
#define DEVICE __device__
#define FDEVICE __forceinline__ DEVICE
#define THREAD_FENCE() __threadfence()
#define CONSTANT __constant__
#define INLINE
#define RESTRICT

#define bDim blockDim
#define tIdx threadIdx
#define bIdx blockIdx

template <class T> FDEVICE T atomicAdd2(T* t, T v) { return atomicAdd(t, v); }
template <class T> FDEVICE T atomicCAS2(T* lk, T old, T nw) { return atomicCAS(lk, old, nw); }

#else

#define GLOBAL
#define DEVICE
#define FDEVICE
#define THREAD_FENCE()
#define CONSTANT
#define INLINE inline
#define RESTRICT __restrict

static uint4 cpu_blockDim;
static uint4 cpu_threadIdx;
static uint4 cpu_blockIdx;
#define bDim cpu_blockDim
#define tIdx cpu_threadIdx
#define bIdx cpu_blockIdx

template <typename T> T atomicAdd2(T* t, T v) { T old = *t; *t += v; return old; }
template <typename T> T atomicCAS2(T* lk, T old, T nw) { *lk = nw; return old; }

#endif


namespace Reylax
{
    template <class T> T min(T a, T b) { return a<b?a:b; }
    template <class T> T max(T a, T b) { return a>b?a:b; }

    FDEVICE INLINE u32 rgb(float r, float g, float b)
    {
    #if RL_CUDA
        u32 ru = lrintf(max(0.f, min(255.f, r*255.f)));
        u32 gu = lrintf(max(0.f, min(255.f, g*255.f)));
        u32 bu = lrintf(max(0.f, min(255.f, b*255.f)));
    #else 
        u32 ru = u32(max(0.f, min(255.f, r*255.f)));
        u32 gu = u32(max(0.f, min(255.f, g*255.f)));
        u32 bu = u32(max(0.f, min(255.f, b*255.f)));
    #endif
        return (ru<<16)|(gu<<8)|bu;
    }

    FDEVICE INLINE u32 rgb(vec3 v)
    {
        return rgb(v.x, v.y, v.z);
    }

    //FDEVICE INLINE float faceRayIntersect(rlFace* face, const vec3& eye, const vec3& dir, const rlMeshData* meshData, float& u, float& v)
    //{
    //    assert(meshDataPtrs);
    //    uint4 idx = face->m_index;
    //    const rlMeshData* mesh = &meshData[idx.w];
    //    vec3* vp  = (vec3*)mesh->m_vertexData[VERTEX_DATA_POSITION];
    //    assert(mesh->m_vertexDataSizes[VERTEX_DATA_POSITION] == 3);
    //    return bmTriIntersect(eye, dir, vp[idx.x], vp[idx.y], vp[idx.z], u, v);
    //}

    FDEVICE INLINE vec4 faceInterpolate(rlFace* face, float u, float v, const rlMeshData* meshData, u32 dataIdx)
    {
        assert(meshData);
        assert(dataIdx < VERTEX_DATA_COUNT);
        const rlMeshData* mesh = &meshData[face->w];
        float* vd = mesh->m_vertexData[dataIdx];
        u32 dsize = mesh->m_vertexDataSizes[dataIdx];
        assert(dsize > 0 && dsize < 4);
        float* vd1 = vd + face->x*dsize;
        float* vd2 = vd + face->y*dsize;
        float* vd3 = vd + face->z*dsize;
        float w = 1-(u+v);
        switch ( dsize )
        {
        case 1:
            return vec4(vd1[0]*u + vd2[0]*v + vd3[0]*w, 0.f, 0.f, 0.f);
        case 2:
            return vec4(vd1[0]*u + vd2[0]*v + vd3[0]*w,
                        vd1[1]*u + vd2[1]*v + vd3[1]*w, 0.f, 0.f);
        case 3:
            return vec4(vd1[0]*u + vd2[0]*w + vd3[0]*w,
                        vd1[1]*u + vd2[1]*w + vd3[1]*w,
                        vd1[2]*u + vd2[2]*w + vd3[2]*w, 0.f);
        case 4:
            return vec4(vd1[0]*u + vd2[0]*v + vd3[0]*w,
                        vd1[1]*u + vd2[1]*v + vd3[1]*w,
                        vd1[2]*u + vd2[2]*v + vd3[2]*w,
                        vd1[3]*u + vd2[3]*v + vd3[3]*w);
        }
        return vec4(0.f);
    }
}
