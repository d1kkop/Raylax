#include "Reylax.h"
// STL
#include <cstdio>
#include <cassert>
#include <iostream>
#include <chrono>
// CDUA
#define CUDA_VERSION 9020
#include <cuda_runtime.h>
// GLM
#include "glm/common.hpp"
#include "glm/geometric.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include "glm/mat3x3.hpp"
#include "glm/mat4x4.hpp"
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/transform.hpp"
#include "glm/gtx/rotate_vector.hpp"
using namespace std;
using namespace glm;
using namespace Reylax;
using namespace chrono;

__device__ u32* buffer;
__device__ void TraceCallback(u32 globalId, u32 localId, const HitResult& hit, const MeshData* const* meshPtrs, float* rayOris, float* rayDirs)
{
    buffer[ globalId ] = 200<<8;
}