#pragma once
#include "ReylaxCuda.h"
#include <iostream>
#include <functional>

#define RL_CUDA_CALL( expr ) \
{ \
	if ( expr != cudaSuccess )  \
	{ \
		std::cout << "CUDA ERROR: " << (expr) << " " << cudaGetErrorString(expr) << std::endl; \
		assert( 0 ); \
		exit( 1 ); \
	} \
}


namespace Reylax
{
    void forEachFace(const MeshData* md, u32 numMeshDatas, const std::function<void (u32, const u32[3], const vec3[3])>& cb);
}