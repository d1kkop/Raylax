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

        DeviceBuffer* m_query;
    };

    struct TraceResult: public ITraceResult
    {
        TraceResult(u32 numRays);
        virtual ~TraceResult();
        DeviceBuffer* m_result;
    };

    struct Tracer: public ITracer
    {
        Tracer();
        virtual ~Tracer();

        u32 trace( const IGpuStaticScene* scene, const ITraceQuery* query, const ITraceResult** results, u32 numResults ) override;
    };
}
