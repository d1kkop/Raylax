#pragma once
#include "ReylaxCuda.h"
#include <iostream>
#include <functional>

#define RL_PRINT_STATS 1

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
    void forEachFace(const MeshData** md, u32 numMeshDatas, const std::function<void (u32, const u32[3], const vec3[3])>& cb);
    void emulateCpu( u32 blockDimension, const dim3& blocks, const dim3& threads, const std::function<void ()>& cb );
    u32  hostOrDeviceCpy( void* dst, const void* src, u32 size, bool srcIsHostData );
}