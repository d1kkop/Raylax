#pragma once
#include "Reylax.h"
#include "ReylaxCommon.h"

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
        u32 m_numRays;
        DeviceBuffer* m_oris;
        DeviceBuffer* m_dirs;
    };

    struct TraceResult: public ITraceResult
    {
        TraceResult(u32 numRays);
        virtual ~TraceResult();
        u32 m_numRays;
        DeviceBuffer* m_result;
    };
}
