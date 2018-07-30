#pragma once
#include "Reylax.h"
#include <stdio.h>
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>

#define RL_WINDOWS      1

#define RL_CUDA_CALL( expr ) \
{ \
	auto status=expr; \
	if ( status != cudaSuccess )  \
	{ \
		std::cout << "CUDA ERROR: " << (status) << " " << cudaGetErrorString(status) << std::endl; \
		assert( 0 ); \
		exit( 1 ); \
	} \
}