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


    struct Material
    {
        u32 texture;
    };

    struct Face
    {
        u32 x, y, z, w;
        Material* mat;
    };

    struct MeshData
    {
        float* vertexData[VERTEX_DATA_COUNT];
        u32    vertexDataSizes[VERTEX_DATA_COUNT];
        u32*   indices;
        u32    numVertices;
        u32    numIndices;
        Material* material;
    };

    struct RayFaceHitResult
    {
        float dist, u, v;
        float ro[3];
        float rd[3];
        Face* face;
    };

    using HitCallback = void (*)(const RayFaceHitResult*, const MeshData*, void**);
}