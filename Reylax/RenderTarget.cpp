#include "RenderTarget.h"
#include "../3rdParty/glatter-master/include/glatter/glatter.h"
//#include <cuda_gl_interop.h>
using namespace Reylax;


namespace Reylax
{
    // ----- IRenderTarget ------------------------------------------------------------------------------------------------------

    IRenderTarget* IRenderTarget::createFromGLTBO(u32 tbo, u32 width, u32 height)
    {
        if ( width==0 || height==0 || tbo==0  ) return nullptr;
        auto rt = new RenderTarget( width, height );
    #if RL_CUDA
        RL_CUDA_CALL( cudaGraphicsGLRegisterBuffer( &rt->m_cudaGraphicsRT, tbo, cudaGraphicsMapFlagsWriteDiscard ) );
        //RL_CUDA_CALL( cudaGraphicsGLRegisterImage( &rt->m_cudaGraphicsRT, openGLRTId, GL_TEXTURE_BUFFER, cudaGraphicsMapFlagsNone ) );
    #else
        rt->m_glTbo = tbo;
    #endif
        return rt;
    }


    // ----- RenderTarget ------------------------------------------------------------------------------------------------------

    RenderTarget::RenderTarget(u32 width, u32 height):
        m_buffer(nullptr),
        m_width(width),
        m_height(height),
        m_cudaGraphicsRT(nullptr)
    {
    }

    RenderTarget::~RenderTarget()
    {
        if ( m_buffer ) unlock();
    #if RL_CUDA
        if ( m_cudaGraphicsRT )
        {
            RL_CUDA_CALL(cudaGraphicsUnregisterResource(m_cudaGraphicsRT));
        }    
    #endif
    }

    u32 RenderTarget::lock()
    {
        assert(!m_buffer);
    #if RL_CUDA
        void* devPtr;
        u64 size;
        if ( m_buffer ) return ERROR_UNLOCK_FIRST;
        RL_CUDA_CALL( cudaGraphicsMapResources( 1, &m_cudaGraphicsRT ) );
        RL_CUDA_CALL( cudaGraphicsResourceGetMappedPointer( &devPtr, &size, m_cudaGraphicsRT ) );
        m_buffer = devPtr;
    #else
        glBindBuffer(GL_TEXTURE_BUFFER, (GLuint) m_glTbo );
        m_buffer = glMapBuffer( GL_TEXTURE_BUFFER, GL_WRITE_ONLY );
    #endif
        return ERROR_ALL_FINE;
    }

    u32 RenderTarget::unlock()
    {
        assert( m_buffer );
    #if RL_CUDA
        if ( !m_buffer ) return ERROR_LOCK_FIRST;
        RL_CUDA_CALL( cudaGraphicsUnmapResources( 1, &m_cudaGraphicsRT ) );
    #else
        glUnmapBuffer( GL_TEXTURE_BUFFER );
    #endif
        m_buffer = nullptr;
        return ERROR_ALL_FINE;
    }
}
