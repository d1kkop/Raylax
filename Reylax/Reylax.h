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

        virtual void* buffer() const = 0;
        virtual u32 width() const  = 0;
        virtual u32 height() const = 0;
        virtual u32 lock() = 0;
        virtual u32 unlock() = 0;
        virtual u32 clear( u32 clearValue ) = 0;
        template <class T> T* buffer() const { return reinterpret_cast<T*>(buffer()); }
    };


    RL_DLL void rlSetDevice(u32 i);
    RL_DLL u32  rlGetNumDevices();
    RL_DLL void rlSyncDevice();
}