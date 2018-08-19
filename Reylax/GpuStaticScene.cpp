#include "GpuStaticScene.h"
#include "DeviceBuffer.h"
#include "Mesh.h"
#include <iostream>
using namespace std;


namespace Reylax
{
    // ------ GpuStaticMesh ----------------------------------------------------------------------------------------


    GpuStaticMesh::~GpuStaticMesh()
    {
        delete d;
        delete indices;
        for ( auto& vd : vertexDatas ) delete vd;
    }

    // ------ GpuStaticScene ----------------------------------------------------------------------------------------

    IGpuStaticScene* IGpuStaticScene::create(const IMesh* const* meshes, u32 numMeshes)
    {
        if ( !meshes || numMeshes==0 )
            return nullptr;

        // Put meshData in new array for easier acces in building the gpu tree.
        vector<const MeshData*> mds;
        for ( u32 i=0; i<numMeshes; i++ )
        {
            const Mesh* m = static_cast<const Mesh*>(meshes[i]);
            // If either of these are zero, then a GPU version is pointless
            if ( !m->d.indices || !m->d.vertexData[VERTEX_DATA_POSITION] )
            {
                return nullptr;
            }
            mds.push_back(&m->d);
        }

        // Create gpu static scene.
        GpuStaticScene* gpuScene = new GpuStaticScene{};

        // Build BVH.
        u32 err = BvhNode::build(mds.data(), (u32)mds.size(),
                                 &gpuScene->m_bvhTree,
                                 &gpuScene->m_faces,
                                 &gpuScene->m_faceClusters,
                                 &gpuScene->m_sides,
                                  gpuScene->bMin,
                                  gpuScene->bMax);

        if ( err != ERROR_ALL_FINE )
        {
            delete gpuScene;
            return nullptr;
        }

        // Convert cpu meshes to gpu versions and make an array of meshPtrs to access them from index 0 to x.
        gpuScene->m_gpuMeshes = new GpuStaticMesh[numMeshes]{};
        for ( u32 i=0; i<numMeshes; ++i )
        {
            const MeshData* md = mds[i];
            GpuStaticMesh* gm  = &gpuScene->m_gpuMeshes[i];

            gm->d = new DeviceBuffer(sizeof(MeshData));
            gm->d->copyFrom(md, false); // From this copy, the correct num indices/vertices and vertex data sizes are set. The valid ptrs are copied below.

            // Gpu indices
            gm->indices = new DeviceBuffer(sizeof(u32)*md->numIndices);
            gm->indices->copyFrom(md->indices, false);
            COPY_PTR_TO_DEVICE_ASYNC(gm->d, gm->indices, MeshData, indices);

            for ( u32 i=0; i<VERTEX_DATA_COUNT; ++i )
            {
                float* ptrAsValue = nullptr;
                if ( md->vertexData[i] )
                {
                    u32 numComponents  = md->vertexDataSizes[i];
                    gm->vertexDatas[i] = new DeviceBuffer(sizeof(float)*numComponents*md->numVertices);
                    gm->vertexDatas[i]->copyFrom( md->vertexData[i], false );
                    ptrAsValue = gm->vertexDatas[i]->ptr<float>();
                }
                else gm->vertexDatas[i] = nullptr;
                COPY_VALUE_TO_DEVICE_ASYNC(gm->d, ptrAsValue, MeshData, vertexData[i], sizeof(float*));
            }
        }

        // Build mesh ptrs for accessing vertex data ultimately at the face level.
        gpuScene->m_meshDataPtrs  = new DeviceBuffer(sizeof(MeshData*)*numMeshes);
        MeshData** deviceMeshPtrs = new MeshData*[numMeshes]; // host data to hold array of device ptrs
        for ( u32 i=0; i<numMeshes; ++i )
        {
            MeshData* dPtr = gpuScene->m_gpuMeshes[i].d->ptr<MeshData>();
            deviceMeshPtrs[i] = dPtr;
        }
        gpuScene->m_meshDataPtrs->copyFrom( deviceMeshPtrs, true);
        delete [] deviceMeshPtrs;
        return gpuScene;
    }

    GpuStaticScene::~GpuStaticScene()
    {
        delete m_bvhTree;
        delete m_faces;
        delete m_faceClusters;
        delete m_sides;
        delete m_meshDataPtrs;
        delete m_gpuMeshes;
    }
}