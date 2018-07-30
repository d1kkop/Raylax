#include "Reylax.h"
#include "Reylax_internal.h"

// For now only CUDA implementation. 
// Perhaps OpenCL, later too.

namespace Reylax
{
    void rlSetDevice(u32 device)
    {
        RL_CUDA_CALL(cudaSetDevice(device));
    }

    u32 rlGetNumDevices()
    {
        i32 count=0;
        RL_CUDA_CALL(cudaGetDeviceCount(&count));
        return count;
    }

    void rlSyncDevice()
    {
        RL_CUDA_CALL(cudaDeviceSynchronize());
    }
    
}