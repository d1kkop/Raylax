#include "SmallGpuStructs.h"
#include "DeviceBuffer.h"
#include "Mesh.h"
#include "ReylaxCommon.h"
#include <iostream>
#include <vector>
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
        printf("TraceQuery numRays: %d, size %fmb\n", numRays, (float)sizeof(float)*3*numRays/1024/1024);
    #endif
        return query;
    }

    TraceQuery::TraceQuery(const float* rays3, u32 numRays):
        m_query(new DeviceBuffer(sizeof(float)*3*numRays))
    {
        m_query->copyFrom(rays3, false);
    }

    TraceQuery::~TraceQuery()
    {
        delete m_query;
    }

    // ------ TraceResult ----------------------------------------------------------------------------------------

    ITraceResult* ITraceResult::create(u32 numRays)
    {
        if (numRays==0) return nullptr;
        ITraceResult* res = new TraceResult(numRays);
        return res;
    }

    TraceResult::TraceResult(u32 numRays):
        m_result(new DeviceBuffer(numRays*sizeof(RayFaceHitResult)))
    {
    #if RL_PRINT_STATS
        printf("TraceResult numRays: %d, size %fmb\n", numRays, (float)sizeof(RayFaceHitResult)*numRays/1024/1024);
    #endif
    }

    TraceResult::~TraceResult()
    {
        delete m_result;
    }

    // ------ Tracer ----------------------------------------------------------------------------------------

    ITracer* ITracer::create()
    {
        return new Tracer();
    }

    Tracer::Tracer()
    {
    }

    Tracer::~Tracer()
    {
    }

    u32 Tracer::trace(const IGpuStaticScene* scene, const ITraceQuery* query, const ITraceResult** results, u32 numResults)
    {
        if ( !scene || !query || numResults==0 || results==nullptr )
        {
            return ERROR_INVALID_PARAMETER;
        }

        for ( u32 i=0; i<numResults; ++i )
            if ( !results[i] ) return ERROR_INVALID_PARAMETER;



        return ERROR_ALL_FINE;
    }

}
