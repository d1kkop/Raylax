#pragma once
#include "Reylax.h"
#include "ReylaxCommon.h"

namespace Reylax
{
    struct DeviceBuffer;

    struct GpuStaticMesh
    {
        ~GpuStaticMesh();
        DeviceBuffer* d{}; // interpetable as MeshData on gpu
        DeviceBuffer* indices{};
        DeviceBuffer* vertexDatas[VERTEX_DATA_COUNT]{};
    };

    struct GpuStaticScene: public IGpuStaticScene
    {
        virtual ~GpuStaticScene();
        DeviceBuffer* m_bvhTree{};
        DeviceBuffer* m_faces{};
        DeviceBuffer* m_faceClusters{};
        DeviceBuffer* m_sides{};
        DeviceBuffer* m_meshDataPtrs{};
        GpuStaticMesh* m_gpuMeshes{};
        vec3 bMin, bMax;
    };
}
