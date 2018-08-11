#include "SmallGpuStructs.h"
#include "DeviceBuffer.h"
#include "Mesh.h"
#include <iostream>
using namespace std;


namespace Reylax
{
    // ------ GpuStaticMesh ----------------------------------------------------------------------------------------

    GpuStaticMesh::GpuStaticMesh()
    {
        memset(this, 0, sizeof(this));
    }

    GpuStaticMesh::~GpuStaticMesh()
    {
        delete d;
        delete indices;
        for ( auto& d : vertexDatas ) delete d;
    }

    // ------ GpuStaticScene ----------------------------------------------------------------------------------------

    IGpuStaticScene* IGpuStaticScene::create(IMesh** meshes, u32 numMeshes)
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
        GpuStaticScene* gpuScene = new GpuStaticScene();

        // Build BVH.
        u32 err = BvhNode::build(mds.data(), (u32)mds.size(),
                                 &gpuScene->m_bvhTree,
                                 &gpuScene->m_faces,
                                 &gpuScene->m_faceClusters);

        if ( err != ERROR_ALL_FINE )
        {
            delete gpuScene;
            return nullptr;
        }

        gpuScene->m_gpuMeshes = new GpuStaticMesh[numMeshes];
        for ( u32 i=0; i<numMeshes; ++i )
        {
            const Mesh* m = static_cast<const Mesh*>(meshes[i]);
            GpuStaticMesh* gm = &gpuScene->m_gpuMeshes[i];

            gm->d = new DeviceBuffer(sizeof(MeshData));
            gm->d->copyFrom(&m->d, false); // From this copy, the correct num indices/vertices and vertex data sizes are set. The valid ptrs are copied below.

            // Gpu indices
            gm->indices = new DeviceBuffer(sizeof(u32)*m->d.numIndices);
            gm->indices->copyFrom(m->d.indices, false);
            COPY_PTR_TO_DEVICE_ASYNC(gm->d, gm->indices, MeshData, indices);

            for ( u32 i=0; i<VERTEX_DATA_COUNT; ++i )
            {
                if ( m->d.vertexData[i] )
                {
                    u32 numComponents  = m->d.vertexDataSizes[i];
                    gm->vertexDatas[i] = new DeviceBuffer(sizeof(float)*numComponents*m->d.numVertices);
                    COPY_PTR_TO_DEVICE_ASYNC(gm->d, gm->vertexDatas[i], MeshData, vertexData[i]);
                }
            }
        }

        // Build mesh ptrs for accessing vertex data ultimately at the face level.
        gpuScene->m_meshDataPtrs = new DeviceBuffer(sizeof(MeshData*)*numMeshes);
        MeshData** meshPtrs      = new MeshData*[numMeshes];
        for ( u32 i=0; i<numMeshes; ++i )
        {
            meshPtrs[i] = gpuScene->m_gpuMeshes[i].d->ptr<MeshData>();
        }
        gpuScene->m_meshDataPtrs->copyFrom(meshPtrs, true);
        delete [] meshPtrs;

        return gpuScene;
    }

    GpuStaticScene::GpuStaticScene():
        m_bvhTree(nullptr),
        m_faces(nullptr),
        m_faceClusters(nullptr),
        m_meshDataPtrs(nullptr),
        m_gpuMeshes(nullptr)
    {
    }

    GpuStaticScene::~GpuStaticScene()
    {
        delete m_bvhTree;
        delete m_faces;
        delete m_faceClusters;
        delete m_meshDataPtrs;
        delete m_gpuMeshes;
    }

    // ------ TraceQuery ----------------------------------------------------------------------------------------

    ITraceQuery* ITraceQuery::create(const float* rays3, u32 numRays)
    {
        if ( !rays3 || numRays==0 ) return nullptr;
        ITraceQuery* query = new TraceQuery(rays3, numRays);
    #if RL_PRINT_STATS
        printf("TraceQuery numRays: %d, size %.3fmb\n", numRays, (float)sizeof(float)*3*numRays/1024/1024);
    #endif
        return query;
    }

    TraceQuery::TraceQuery(const float* rays3, u32 numRays):
        m_numRays(numRays),
        m_oris(new DeviceBuffer(sizeof(float)*3*numRays)), /* oris are for secondary ray launches */
        m_dirs(new DeviceBuffer(sizeof(float)*3*numRays))
    {
        assert(numRays!=0);
        m_dirs->copyFrom(rays3, false);
    }

    TraceQuery::~TraceQuery()
    {
        delete m_oris;
        delete m_dirs;
    }

    // ------ TraceResult ----------------------------------------------------------------------------------------

    ITraceResult* ITraceResult::create(u32 numRays)
    {
        if (numRays==0) return nullptr;
        ITraceResult* res = new TraceResult(numRays);
        return res;
    }

    TraceResult::TraceResult(u32 numRays):
        m_numRays(numRays),
        m_result(new DeviceBuffer(numRays*sizeof(HitResult)))
    {
        assert(m_numRays!=0);
    #if RL_PRINT_STATS
        printf("TraceResult numRays: %d, size %.3fmb\n", numRays, (float)sizeof(HitResult)*numRays/1024/1024);
    #endif
    }

    TraceResult::~TraceResult()
    {
        delete m_result;
    }

}