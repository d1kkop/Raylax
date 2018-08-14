#pragma once
#include "Reylax.h"
#include "ReylaxCommon.h"
#include "Reylax_internal.h"


namespace Reylax
{
    struct Tracer: public ITracer
    {
        Tracer(u32 numPointBoxQueries=256*256*64, u32 numLeafQueries=256*256*64, u32 numRaysPerTile=256*256);
        virtual ~Tracer();
        u32 trace(const float* eye3, const float* orient3x3,
                  const IGpuStaticScene* scene, const ITraceQuery* query, 
                  const ITraceResult* const* results, u32 numResults,
                  HitCallback cb) override;

        DeviceBuffer* m_pointBoxQueue[2];
        DeviceBuffer* m_leafQueue[2];
        DeviceBuffer* m_pointBoxBuffer[2];
        DeviceBuffer* m_leafBuffer[2];
        u32 m_numPointBoxQueries;
        u32 m_numRayLeafQueries;
        u32 m_numRaysPerTile;
        Profiler m_profiler;
    };
}
