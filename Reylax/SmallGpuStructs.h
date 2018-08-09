#pragma once
#include "Reylax.h"

namespace Reylax
{
    struct DeviceBuffer;

    struct GpuStaticScene: public IGpuStaticScene
    {
        GpuStaticScene();
        virtual ~GpuStaticScene();

        DeviceBuffer* m_bvhTree;
        DeviceBuffer* m_faces;
        DeviceBuffer* m_faceClusters;
    };

    struct TraceQuery: public ITraceQuery
    {
        TraceQuery(const float* rays3, u32 numRays);
        virtual ~TraceQuery();

        DeviceBuffer* m_oris;
        DeviceBuffer* m_dirs;
    };

    struct TraceResult: public ITraceResult
    {
        TraceResult(u32 numRays);
        virtual ~TraceResult();
        DeviceBuffer* m_result;
    };

    struct Tracer: public ITracer
    {
        Tracer(u32 numRayBoxQueries=256*256*256, u32 numLeafQueries=256*256*256);
        virtual ~Tracer();
        void resetRayBoxQueue( u32 idx );
        void resetRayLeafQueue();
        u32 trace( const float* eye3, const float* orient3x3, const IGpuStaticScene* scene, const ITraceQuery* query, const ITraceResult* const* results, u32 numResults ) override;

        DeviceBuffer* m_rayBoxQueue[2];
        DeviceBuffer* m_leafQueue;
        DeviceBuffer* m_rayBoxBuffer[2];
        DeviceBuffer* m_leafBuffer;
        u32 m_numRayBoxQueries;
        u32 m_numRayLeafQueries;
    };
}
