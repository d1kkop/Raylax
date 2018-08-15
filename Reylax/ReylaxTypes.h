#pragma once


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


    __declspec(align(4))
    struct Material
    {
        u32 texture;
    };

    __declspec(align(8))
    struct Face
    {
        const Material* mat;
        u32 x, y, z, w;
    };

    __declspec(align(8))
    struct MeshData
    {
        float* vertexData[VERTEX_DATA_COUNT];
        u32*   indices;
        Material* material;
        u32    vertexDataSizes[VERTEX_DATA_COUNT];
        u32    numVertices;
        u32    numIndices;
    };

    __declspec(align(8))
    struct HitResult
    {
        const Face* face;
        float dist, u, v;
        float ro[3];
        float rd[3];
    };


    using RaySetupFptr      = void (*)(u32 globalId, u32 localId);
    using HitResultFptr     = void (*)(u32 globalId, u32 localId, u32 depth, const HitResult& hit, const MeshData* const* meshPtrs);
    using QueueRayFptr      = void (*)(u32 localId, const float* ori3, const float* dir3);
}