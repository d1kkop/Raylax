#include "SmallGpuStructs.h"
#include "DeviceBuffer.h"
#include "Mesh.h"
#include <iostream>
using namespace std;


namespace Reylax
{
    // ------ GpuStaticScene ----------------------------------------------------------------------------------------

    IGpuStaticScene* IGpuStaticScene::create(IMesh* const* meshes, u32 numMeshes)
    {
        if ( !meshes || numMeshes==0 )
            return nullptr;

        vector<const MeshData*> mds;
        for ( u32 i=0; i<numMeshes; i++ )
        {
            mds.push_back(&static_cast<const Mesh*>(meshes[i])->d);
        }

        GpuStaticScene* gpuScene = new GpuStaticScene();

        u32 err = BvhNode::build(mds.data(), (u32)mds.size(),
                                 &gpuScene->m_bvhTree,
                                 &gpuScene->m_faces,
                                 &gpuScene->m_faceClusters);

        if ( err != ERROR_ALL_FINE )
        {
            cout << "GpuStaticScene creation error: " << err << endl;
            delete gpuScene;
            return nullptr;
        }

        return gpuScene;
    }

    GpuStaticScene::GpuStaticScene():
        m_bvhTree(nullptr),
        m_faces(nullptr),
        m_faceClusters(nullptr)
    {
    }

    GpuStaticScene::~GpuStaticScene()
    {
        delete m_bvhTree;
        delete m_faces;
        delete m_faceClusters;
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
        m_result(new DeviceBuffer(numRays*sizeof(RayFaceHitResult)))
    {
        assert(m_numRays!=0);
    #if RL_PRINT_STATS
        printf("TraceResult numRays: %d, size %.3fmb\n", numRays, (float)sizeof(RayFaceHitResult)*numRays/1024/1024);
    #endif
    }

    TraceResult::~TraceResult()
    {
        delete m_result;
    }

}