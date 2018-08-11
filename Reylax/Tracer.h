#pragma once
#include "Reylax.h"
#include "ReylaxCommon.h"
#include "Reylax_internal.h"


namespace Reylax
{
    struct Tracer: public ITracer
    {
        Tracer(u32 numRayBoxQueries=256*256*64*4, u32 numLeafQueries=256*256*64*8, u32 numRayFaceQueries=256*256*64*32, u32 numRaysPerTile=256*256);
        virtual ~Tracer();
        u32 trace(const float* eye3, const float* orient3x3,
                  const IGpuStaticScene* scene, const ITraceQuery* query, const ITraceResult* const* results, u32 numResults) override;

        DeviceBuffer* m_rayBoxQueue[2];
        DeviceBuffer* m_leafQueue;
        DeviceBuffer* m_rayBoxBuffer[2];
        DeviceBuffer* m_leafBuffer;
        DeviceBuffer* m_hitResultClusters;
        u32 m_numRayBoxQueries;
        u32 m_numRayLeafQueries;
        u32 m_numRayFaceQueries;
        u32 m_numRaysPerTile;
        Profiler m_profiler;
    };
}
