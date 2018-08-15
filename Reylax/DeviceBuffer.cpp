#include "DeviceBuffer.h"
#include "Reylax_internal.h"

namespace Reylax
{
    IDeviceBuffer* IDeviceBuffer::create(u32 size)
    {
        return new DeviceBuffer(size);
    }

    DeviceBuffer::DeviceBuffer(u32 size):
        m_size(size),
        m_devData(nullptr)
    {
        assert(m_size);
    #if RL_CUDA
        RL_CUDA_CALL(cudaMalloc(&m_devData, size));
    #else
        m_devData = malloc(size);
    #endif
    }

    DeviceBuffer::~DeviceBuffer()
    {
    #if RL_CUDA
        RL_CUDA_CALL(cudaFree(m_devData));
    #else
        free(m_devData);
    #endif
    }

    void DeviceBuffer::copyTo(void* buffer, bool wait)
    {
    #if RL_CUDA
        if ( wait )
        {
            RL_CUDA_CALL(cudaMemcpy(buffer, m_devData, m_size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        }
        else
        {
            RL_CUDA_CALL(cudaMemcpyAsync(buffer, m_devData, m_size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        }
    #else
        memcpy( buffer, m_devData, m_size );
    #endif
    }

    void DeviceBuffer::copyFrom(const void* buffer, bool wait)
    {
    #if RL_CUDA
        if ( wait )
        {
            RL_CUDA_CALL(cudaMemcpy(m_devData, buffer, m_size, cudaMemcpyKind::cudaMemcpyHostToDevice));
        }
        else
        {
            RL_CUDA_CALL(cudaMemcpyAsync(m_devData, buffer, m_size, cudaMemcpyKind::cudaMemcpyHostToDevice));
        }
    #else
        memcpy( m_devData, buffer, m_size );
    #endif
    }

}