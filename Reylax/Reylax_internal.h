#pragma once
#include "ReylaxCuda.h"
#include <iostream>
#include <functional>
#include <vector>
#include <string>


#define RL_CUDA_CALL( expr ) \
{ \
	if ( expr != cudaSuccess )  \
	{ \
		std::cout << "CUDA ERROR: " << (expr) << " " << cudaGetErrorString(expr) << std::endl; \
		assert( 0 ); \
		exit( 1 ); \
	} \
}

#if RL_CUDA
    #define RL_KERNEL_CALL( bdim, blocks, threads, name, ... ) \
{\
        name<<<blocks, threads>>>( __VA_ARGS__ );\
        auto err = cudaGetLastError(); \
        RL_CUDA_CALL( err ); \
}

#else
    #define RL_KERNEL_CALL( bdim, blocks, threads, name, ... ) \
            Reylax::emulateCpu( #name, bdim, blocks, threads, [=]() { \
                name( __VA_ARGS__ ); \
            })
#endif

#if RL_CUDA_DYN
    #define RL_KERNEL_CALL_DYN ( bdim, blocks, threads, name, ... ) \
        name<<<blocks, threads>>>( __VA_ARGS__ )
#else
    #define RL_KERNEL_CALL_DYN( bdim, blocks, threads, name, ... ) \
            Reylax::emulateCpu( #name, bdim, blocks, threads, [=]() { \
                name( __VA_ARGS__ ); \
            })
#endif

#define DELETE_AND_NULL( p ) delete p; p=nullptr
#define DELETE_AR_AND_NULL( p ) delete [] p; p=nullptr;
#define COPY_VALUE_TO_DEVICE_ASYNC( dst, value, str, member, size ) Reylax::hostOrDeviceCpy( dst->ptr<char>() + offsetof(str, member), &value, size, cudaMemcpyHostToDevice, true )
#define COPY_PTR_TO_DEVICE_ASYNC( dst, src, str, member ) \
{\
    u64 pVal = (u64)(src)->ptr<char>(); \
    COPY_VALUE_TO_DEVICE_ASYNC( (dst), pVal, str, member, sizeof(char*)); \
}


namespace Reylax
{
    double time(); 
    void forEachFace(const MeshData** md, u32 numMeshDatas, const std::function<void (u32, const u32[3], const vec3[3])>& cb);
    void emulateCpu( const char* profileName, u32 blockDimension, const dim3& blocks, const dim3& threads, const std::function<void ()>& cb );
    u32  hostOrDeviceCpy( void* dst, const void* src, u32 size, cudaMemcpyKind kind, bool async );


    template <typename T>
    void SetSymbol(T& dst, const void* src, bool wait=false)
    {
    #if RL_CUDA
        if ( !wait ) { RL_CUDA_CALL(cudaMemcpyToSymbolAsync(dst, src, sizeof(T), 0, cudaMemcpyDefault)); }
        else { RL_CUDA_CALL(cudaMemcpyToSymbol(dst, src, sizeof(T), 0, cudaMemcpyDefault)); }
    #else
        memcpy(&dst, src, sizeof(T));
    #endif
    }

    template <typename T>
    void GetSymbol(void* dst, const T& src, bool wait=true)
    {
    #if RL_CUDA
        if ( !wait ) { RL_CUDA_CALL(cudaMemcpyFromSymbolAsync(dst, src, sizeof(T), 0, cudaMemcpyDeviceToHost)); }
        else { RL_CUDA_CALL(cudaMemcpyFromSymbol(dst, src, sizeof(T), 0, cudaMemcpyDeviceToHost)); }
    #else
        memcpy(dst, &src, sizeof(T));
    #endif
    }


    struct Profiler
    {
        double m_interval=2000;
        bool m_show=false;
        double m_start;
        std::vector<std::pair<std::string, double>> m_items;

        void setInterval(u32 ms) { m_interval = ms; }
        void beginProfile() { static double t=time(); double tn=time(); if ( (tn-t)>m_interval ) { m_show=true; t=tn; } }
        void start() { if(!m_show) return; m_start = time(); }
        void stop(const std::string& name) { if (!m_show)return; m_items.emplace_back(name, time()-m_start); }
        void endProfile(const std::string& header)
        {
            if (!m_show) return;
            printf("---- %s ---- \n", header.c_str());
            for ( auto& i : m_items ) printf("%s:\t\t%.3fms\n", i.first.c_str(), i.second);
            m_items.clear();
            m_show=false;
        }
    };
}