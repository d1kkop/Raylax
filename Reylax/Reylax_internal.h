#pragma once
#include "Cuda.h"
#include <iostream>

#define RL_CUDA_CALL( expr ) \
{ \
	if ( expr != cudaSuccess )  \
	{ \
		std::cout << "CUDA ERROR: " << (expr) << " " << cudaGetErrorString(expr) << std::endl; \
		assert( 0 ); \
		exit( 1 ); \
	} \
}
