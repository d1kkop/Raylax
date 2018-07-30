#pragma once

#if RL_IMPORT
    #define RL_DLL __declspec(dllimport)
#elif  RL_EXPORT 
    #define RL_DLL __declspec(dllexport)
#else
    #define RL_DLL
#endif

namespace Reylax
{
    

    using i32 = __int32;
    using i64 = __int64;
    using u32 = unsigned __int32;
    using u64 = unsigned __int64;


    constexpr u32 ERROR_ALL_FINE            = 0;
    constexpr u32 ERROR_NO_VERTICES         = 1;
    constexpr u32 ERROR_INVALID_PARAMETER   = 2;
    constexpr u32 ERROR_GPU_ALLOC_FAIL      = 3;
    constexpr u32 ERROR_INVALID_FORMAT      = 4;
    constexpr u32 ERROR_RT_CAM_MISMATCH     = 5;
    constexpr u32 ERROR_UNLOCK_FIRST        = 6;
    constexpr u32 ERROR_LOCK_FIRST          = 7;
    constexpr u32 ERROR_NO_RENDER_TARGET    = 8;


    constexpr u32 VERTEX_DATA_POSITION      = 0;
    constexpr u32 VERTEX_DATA_NORMAL        = 1;
    constexpr u32 VERTEX_DATA_UV1           = 2;
    constexpr u32 VERTEX_DATA_UV2           = 3;
    constexpr u32 VERTEX_DATA_TANGENT       = 4;
    constexpr u32 VERTEX_DATA_BITANGENT     = 5;
    constexpr u32 VERTEX_DATA_EXTRA1        = 6;
    constexpr u32 VERTEX_DATA_EXTRA2        = 7;
    constexpr u32 VERTEX_DATA_EXTRA3        = 8;
    constexpr u32 VERTEX_DATA_EXTRA4        = 9;
    constexpr u32 VERTEX_DATA_COUNT         = 10; // This is not a slot



    struct IRenderTarget
    {
        // Create render target rom OpenGL Texture Buffer Object.
        RL_DLL static IRenderTarget* createFromGLTBO(u32 rtId, u32 width, u32 height);

        virtual void* buffer() const = 0;
        virtual u32 width() const  = 0;
        virtual u32 height() const = 0;
        virtual u32 lock() = 0;
        virtual u32 unlock() = 0;

        template <class T> T* buffer() const { return reinterpret_cast<T*>(buffer()); }
    };
}