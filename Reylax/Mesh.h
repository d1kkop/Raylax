#pragma once
#include "Reylax.h"

namespace Reylax
{
    struct DeviceBuffer;

    class Mesh: public IMesh
    {
    public:
        virtual ~Mesh();

        u32 setVertexData(const float* data, u32 numVertices, u32 numComponents, u32 slotId) override;
        u32 setIndices(const u32* indices, u32 numIndices) override;

        u32 numVertices() const { return d.numVertices; }
        u32 numIndices() const { return d.numIndices; }
        u32* indices() const { return d.indices; }
        float* vertexData(u32 idx) const { return d.vertexData[idx]; }
        u32 vertexDataSize(u32 idx) const { return d.vertexDataSizes[idx]; }

        MeshData d;
    };
}
