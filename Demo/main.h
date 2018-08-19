#pragma once

#include "Reylax.h"
#include "GLinterop.h"
// SDL
#include <SDL.h>
// STL
#include <cstdio>
#include <cassert>
#include <iostream>
#include <chrono>
// CDUA
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

#define DEMO_CPU 1

#if !DEMO_CPU
#define HOST_OR_DEVICE __device__
#else
#define HOST_OR_DEVICE
#endif

#define DEMO_NAME "RayDemo"
#define SCREEN_WIDTH 1920
#define SCREEN_HEIGHT 1080

using namespace glm;
using namespace Reylax;

__align__(8)
struct TraceData
{
    vec3 eye;
    mat3 orient;
    vec3* rayDirs;
    u32*  pixels;
};