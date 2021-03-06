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

    struct IDeviceBuffer
    {
        RL_DLL static IDeviceBuffer* create(u32 size);
        virtual ~IDeviceBuffer() = default;
        virtual void copyFrom( const void* src, bool wait=false ) = 0;
        virtual void copyTo( void* dst, bool wait=false ) = 0;
        virtual u32  size() const = 0;
        virtual void* ptr() const = 0;

        template <class T>
        inline T* ptr() const { return reinterpret_cast<T*>(ptr()); }
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

    struct ITracer
    {
        RL_DLL static ITracer* create(u32 numRaysPerTile=1024*1024, u32 maxRecursionDepth=8);
        virtual ~ITracer() = default;
        virtual u32 trace( u32 numRays, const IGpuStaticScene* scene, RaySetupFptr setupCb, HitResultFptr hitCb ) = 0;
        virtual u32 trace2( u32 numRays, const IGpuStaticScene* scene, const float* eye3, const float* orient3x3, 
                            const float* rays3, const u32* pixels ) = 0;
        virtual QueueRayFptr getQueueRayAddress() const = 0;
    };


    RL_DLL void SetDevice(u32 i);
    RL_DLL u32  GetNumDevices();
    RL_DLL void SyncDevice();
}