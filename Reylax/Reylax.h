#pragma once
#include "ReylaxTypes.h"

#if RL_IMPORT
    #define RL_DLL __declspec(dllimport)
#elif  RL_EXPORT 
    #define RL_DLL __declspec(dllexport)
#else
    #define RL_DLL
#endif

namespace Reylax
{
    constexpr u32 ERROR_ALL_FINE            = 0;
    constexpr u32 ERROR_NO_VERTICES         = 1;
    constexpr u32 ERROR_INVALID_PARAMETER   = 2;
    constexpr u32 ERROR_GPU_ALLOC_FAIL      = 3;
    constexpr u32 ERROR_INVALID_FORMAT      = 4;
    constexpr u32 ERROR_RT_CAM_MISMATCH     = 5;
    constexpr u32 ERROR_UNLOCK_FIRST        = 6;
    constexpr u32 ERROR_LOCK_FIRST          = 7;
    constexpr u32 ERROR_NO_RENDER_TARGET    = 8;


    struct IRenderTarget
    {
        // Create render target rom OpenGL Texture Buffer Object.
        RL_DLL static IRenderTarget* createFromGLTBO(u32 rtId, u32 width, u32 height);
        virtual ~IRenderTarget() = default;
        virtual void* buffer() const = 0;
        virtual u32 width() const  = 0;
        virtual u32 height() const = 0;
        virtual u32 lock() = 0;
        virtual u32 unlock() = 0;
        virtual u32 clear( u32 clearValue ) = 0;
        template <class T> T* buffer() const { return reinterpret_cast<T*>(buffer()); }
    };

    struct IMesh
    {
        RL_DLL static IMesh* create();
        virtual ~IMesh() = default;
        virtual u32 setVertexData(const float* vertices, u32 numVertices, u32 numComponents, u32 slotId) = 0;
        virtual u32 setIndices(const u32* indices, u32 numIndices) = 0;
    };

    struct IGpuStaticScene
    {
        RL_DLL static IGpuStaticScene* create(const IMesh* const* meshes, u32 numMeshes);
        virtual ~IGpuStaticScene() = default;
    };

    struct ITraceQuery
    {
        RL_DLL static ITraceQuery* create(const float* rays3, u32 numRays);
        virtual ~ITraceQuery() = default;
    };

    struct ITraceResult
    {
        RL_DLL static ITraceResult* create(u32 numRays);
        virtual ~ITraceResult() = default;
    };

    struct ITracer
    {
        RL_DLL static ITracer* create(u32 numRaysPerTile=256*256, u32 maxRecursionDepth=8);
        virtual ~ITracer() = default;
        virtual u32 trace( u32 numRays, const IGpuStaticScene* scene, RaySetupFptr setupCb, HitResultFptr hitCb ) = 0;
        virtual QueueRayFptr getQueueRayAddress() const = 0;
    };


    RL_DLL void setDevice(u32 i);
    RL_DLL u32  getNumDevices();
    RL_DLL void syncDevice();
    RL_DLL void setSymbolPtrAsync( const void* dst, const void* src );
}