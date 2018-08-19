#pragma once
#include "Reylax.h"
#include "ReylaxCommon.h"
#include "Reylax_internal.h"


namespace Reylax
{
    struct Tracer: public ITracer
    {
        Tracer(u32 numRaysPerTile=256*256, u32 maxRecursionDepth=8);
        virtual ~Tracer();
        u32 trace( u32 numRays, const IGpuStaticScene* scene, RaySetupFptr setupCb, HitResultFptr hitCb ) override;
        QueueRayFptr getQueueRayAddress() const override;

        DeviceBuffer* m_pointBoxQueue[2]{};
        DeviceBuffer* m_leafQueue[2]{};
        DeviceBuffer* m_pointBoxBuffer[2]{};
        DeviceBuffer* m_leafBuffer[2]{};
        DeviceBuffer* m_rayQueue{};
        DeviceBuffer* m_rayBuffer{};
        DeviceBuffer* m_hitResults{};
        Profiler m_profiler{};
        TracerContext m_ctx{};
        u32 m_numRaysPerTile{};
    };
}
