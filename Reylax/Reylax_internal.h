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
        name<<<blocks, threads>>>( __VA_ARGS__ )
#else
#define RL_KERNEL_CALL( bdim, blocks, threads, name, ... ) \
            Reylax::emulateCpu( bdim, blocks, threads, [=](){ \
                name( __VA_ARGS__ ); \
            })
#endif


namespace Reylax
{
    double time(); 
    void forEachFace(const MeshData** md, u32 numMeshDatas, const std::function<void (u32, const u32[3], const vec3[3])>& cb);
    void emulateCpu( u32 blockDimension, const dim3& blocks, const dim3& threads, const std::function<void ()>& cb );
    u32  hostOrDeviceCpy( void* dst, const void* src, u32 size, cudaMemcpyKind kind, bool async );


    struct Profiler
    {
        double m_start;
        std::vector<std::pair<std::string, double>> m_items;
        double m_interval;
        bool m_show;

        Profiler():
            m_show(true),
            m_interval(2000)
        {
        }
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