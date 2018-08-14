#pragma once
#include "Reylax.h"
#include "ReylaxCommon.h"
#include "Reylax_internal.h"


namespace Reylax
{
    struct Tracer: public ITracer
    {
        Tracer(u32 numPointBoxQueries=256*256*4, u32 numLeafQueries=256*256*4, u32 numRayQueries=256*256*5, u32 numRaysPerTile=256*256);
        virtual ~Tracer();
        u32 trace( u32 numRays, const IGpuStaticScene* scene, RaySetupFptr setupCb, HitResultFptr hitCb ) override;
        QueueRayFptr getQueueRayAddress() const override;

        DeviceBuffer* m_pointBoxQueue[2];
        DeviceBuffer* m_leafQueue[2];
        DeviceBuffer* m_pointBoxBuffer[2];
        DeviceBuffer* m_leafBuffer[2];
        DeviceBuffer* m_rayQueue;
        DeviceBuffer* m_rayBuffer;
        u32 m_numPointBoxQueries;
        u32 m_numRayLeafQueries;
        u32 m_numRayQueries;
        u32 m_numRaysPerTile;
        Profiler m_profiler;
    };
}
