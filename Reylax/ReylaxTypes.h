#pragma once

#define RL_CUDA 1

namespace Reylax
{
    using i32 = __int32;
    using i64 = __int64;
    using u32 = unsigned __int32;
    using u64 = unsigned __int64;

    constexpr u32 VERTEX_DATA_POSITION      = 0;
    constexpr u32 VERTEX_DATA_NORMAL        = 1;
    constexpr u32 VERTEX_DATA_UV1           = 2;
    constexpr u32 VERTEX_DATA_UV2           = 3;
    constexpr u32 VERTEX_DATA_TANGENT       = 4;
    constexpr u32 VERTEX_DATA_BITANGENT     = 5;
    constexpr u32 VERTEX_DATA_EXTRA1        = 6;
    constexpr u32 VERTEX_DATA_EXTRA2        = 7;
    constexpr u32 VERTEX_DATA_EXTRA3        = 8;
    constexpr u32 VERTEX_DATA_EXTRA4        = 9;
    constexpr u32 VERTEX_DATA_COUNT         = 10; // This is not a slot


    struct rlMaterial
    {
        u32 m_texture;
    };

    struct rlFace
    {
        rlMaterial* m_mat;
        u32 x, y, z, w;
    };

    struct rlMeshData
    {
        float* m_vertexData[VERTEX_DATA_COUNT];
        u32  m_vertexDataSizes[VERTEX_DATA_COUNT];
        u32* m_indices;
        u32  m_numVertices;
        u32  m_numIndices;
    };
}