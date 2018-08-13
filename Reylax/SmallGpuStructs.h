#pragma once
#include "Reylax.h"
#include "ReylaxCommon.h"

namespace Reylax
{
    struct DeviceBuffer;

    struct GpuStaticMesh
    {
        GpuStaticMesh();
        ~GpuStaticMesh();
        DeviceBuffer* d; // interpetable as MeshData on gpu
        DeviceBuffer* indices;
        DeviceBuffer* vertexDatas[VERTEX_DATA_COUNT];
    };

    struct GpuStaticScene: public IGpuStaticScene
    {
        GpuStaticScene();
        virtual ~GpuStaticScene();
        DeviceBuffer* m_bvhTree;
        DeviceBuffer* m_faces;
        DeviceBuffer* m_faceClusters;
        DeviceBuffer* m_sides;
        DeviceBuffer* m_meshDataPtrs;
        GpuStaticMesh* m_gpuMeshes;
    };

    struct TraceQuery: public ITraceQuery
    {
        TraceQuery(const float* rays3, u32 numRays);
        virtual ~TraceQuery();
        u32 m_numRays;
        DeviceBuffer* m_oris;
        DeviceBuffer* m_dirs;
        DeviceBuffer* m_signs;
    };

    struct TraceResult: public ITraceResult
    {
        TraceResult(u32 numRays);
        virtual ~TraceResult();
        u32 m_numRays;
        DeviceBuffer* m_result;
    };
}
