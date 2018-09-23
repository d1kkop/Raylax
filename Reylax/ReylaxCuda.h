#pragma once
#include "Cuda.h"
#include "glm/common.hpp"
#include "glm/geometric.hpp"
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/transform.hpp"
#include "glm/gtx/rotate_vector.hpp"
#include "ReylaxTypes.h"
using namespace glm;

#define RL_CUDA 1
#define RL_CUDA_DYN 0
#define RL_CPU_MT 0
#define RL_USE_INNER_QUEUES 0
#define RL_NUM_INNER_QUEUES 4

#define RL_PRINT_STATS 1
#define RL_RAY_ITERATIONS 32

#if RL_CUDA

#define GLOBAL __global__
#define DEVICE __device__
#define FDEVICE __forceinline__ DEVICE
#define THREAD_FENCE() __threadfence()
#define CONSTANT __constant__
#define INLINE
#define RESTRICT
#define SHARED __shared__
#define HOST __host__

#if RL_CUDA_DYN
#define GLOBAL_DYN GLOBAL
#define DEVICE_DYN DEVICE
#define FDEVICE_DYN FDEVICE
#else
#define GLOBAL_DYN
#define DEVICE_DYN
#define FDEVICE_DYN
#endif

#define bDim blockDim
#define tIdx threadIdx
#define bIdx blockIdx

template <class T> FDEVICE T atomicAdd2(T* t, T v) { return atomicAdd(t, v); }

#else

#define GLOBAL
#define GLOBAL_DYN
#define DEVICE
#define DEVICE_DYN
#define FDEVICE
#define FDEVICE_DYN
#define THREAD_FENCE()
#define CONSTANT
#define INLINE inline
#define RESTRICT __restrict
#define SHARED
#define HOST

extern uint4 cpu_blockDim;
extern thread_local uint4 cpu_threadIdx;
extern thread_local uint4 cpu_blockIdx;
#define bDim cpu_blockDim
#define tIdx cpu_threadIdx
#define bIdx cpu_blockIdx

#if !RL_CPU_MT
template <typename T> T atomicAdd2(T* t, T v) { T old = *t; *t += v; return old; }
#else
#include <ppl.h>
template <typename T> T atomicAdd2(T* t, T v) { return t->fetch_add(v, std::memory_order_relaxed); }
#endif

#endif


namespace Reylax
{
    template <class T> FDEVICE HOST INLINE T _min(T a, T b) { return a<b?a:b; }
    template <class T> FDEVICE HOST INLINE T _max(T a, T b) { return a>b?a:b; }
    template <> FDEVICE HOST INLINE vec3 _min(vec3 a, vec3 b) { return vec3(_min<float>(a.x, b.x), _min<float>(a.y, b.y), _min<float>(a.z, b.z)); }
    template <> FDEVICE HOST INLINE vec3 _max(vec3 a, vec3 b) { return vec3(_max<float>(a.x, b.x), _max<float>(a.y, b.y), _max<float>(a.z, b.z)); }
}
