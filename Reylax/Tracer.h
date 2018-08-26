#pragma once
#include "Reylax.h"
#include "ReylaxCommon.h"
#include "Reylax_internal.h"


namespace Reylax
{
    struct Tracer: public ITracer
    {
        Tracer(u32 numRaysPerTile=1024*1024, u32 maxRecursionDepth=8);
        virtual ~Tracer();
        u32 trace( u32 numRays, const IGpuStaticScene* scene, RaySetupFptr setupCb, HitResultFptr hitCb ) override;
        u32 trace2(u32 numRays, const IGpuStaticScene* scene, const float* eye3, const float* orient3x3,
                   const float* rays3, const u32* pixels) override;
        QueueRayFptr getQueueRayAddress() const override;

        DeviceBuffer* m_pointBoxQueue[2]{};
        DeviceBuffer* m_leafQueue[2]{};
        DeviceBuffer* m_pointBoxBuffer[2]{};
        DeviceBuffer* m_leafBuffer[2]{};
        DeviceBuffer* m_rayQueue{};
        DeviceBuffer* m_rayBuffer{};
        DeviceBuffer* m_hitResults{};
        DeviceBuffer* m_id2Queue{};
    //    DeviceBuffer* m_id2RayQueue{};
        TracerContext m_ctx{};
        u32 m_numRaysPerTile{};
        struct GpuStaticScene* m_lastTracedScene{};
    };
}
