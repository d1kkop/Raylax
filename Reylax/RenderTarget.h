#pragma once
#include "Reylax.h"
#include <cuda_runtime.h>

namespace Reylax
{
    struct RenderTarget: public IRenderTarget
    {
        RenderTarget(u32 width, u32 height);
        virtual ~RenderTarget();

        u32 lock() override;
        u32 unlock() override;
        void* buffer() const override { return m_buffer; }
        u32 width() const override  { return m_width; }
        u32 height() const override { return m_height; }
        u32 clear( u32 clearValue ) override;

        union
        {
            cudaGraphicsResource* m_cudaGraphicsRT;
            u64 m_glTbo;
        };

    private:
        u32 m_width, m_height;
        void* m_buffer;
    };
}