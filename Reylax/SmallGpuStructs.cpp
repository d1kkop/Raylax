#include "SmallGpuStructs.h"
#include "DeviceBuffer.h"
#include "Mesh.h"
#include "ReylaxCommon.h"
#include "Reylax_internal.h"
#include <iostream>
#include <vector>
using namespace std;


extern "C"
{
    using namespace Reylax;
    u32 rlRayBox(const float* eye3, const float* orient3x3, u32 numRayBoxes, 
                 Store<RayBox>* rayBoxQueueIn, Store<RayBox>* rayBoxQueueOut, Store<u32>* leafQueue, 
                 const vec3* rayDirs, const BvhNode* bvhNodes);
}


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
        m_oris(new DeviceBuffer(sizeof(float)*3*numRays)), /* oris are for secondary ray launches */
        m_dirs(new DeviceBuffer(sizeof(float)*3*numRays))
    {
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
        m_result(new DeviceBuffer(numRays*sizeof(RayFaceHitResult)))
    {
    #if RL_PRINT_STATS
        printf("TraceResult numRays: %d, size %.3fmb\n", numRays, (float)sizeof(RayFaceHitResult)*numRays/1024/1024);
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

    Tracer::Tracer(u32 numRayBoxQueries, u32 numLeafQueries):
        m_numRayBoxQueries(numRayBoxQueries),
        m_numRayLeafQueries(numLeafQueries),
        m_leafQueue(nullptr),
        m_leafBuffer(nullptr)
    {
        for ( u32 i=0; i<2; i++ )
        {
            m_rayBoxQueue[i]  = nullptr;
            m_rayBoxBuffer[i] = nullptr;
        }
        m_leafQueue  = new DeviceBuffer(sizeof(Store<u32>));
        m_leafBuffer = new DeviceBuffer(numLeafQueries*sizeof(u32));

    #if RL_PRINT_STATS
        printf("RayLeafQueries count %d, size %.3fmb\n", numLeafQueries, (float)m_leafBuffer->size()/1024/1024);
    #endif
        for ( u32 i=0; i<2; i++ )
        {
            m_rayBoxQueue[i]  = new DeviceBuffer(sizeof(Store<RayBox>));
            m_rayBoxBuffer[i] = new DeviceBuffer(numRayBoxQueries*sizeof(RayBox));

        #if RL_PRINT_STATS
            printf("RayBoxQueries %d, count %d, size %.3fmb\n", i, numRayBoxQueries, (float)m_rayBoxBuffer[i]->size()/1024/1024);
        #endif
        }
         
        // Assign device buffers to queue elements ptr
        u64 pLeafBuffer = (u64)m_leafBuffer->ptr<void>();
        hostOrDeviceCpy( ((char*)m_leafQueue->ptr<Store<u32>>()) + offsetof(Store<u32>, m_elements), &pLeafBuffer, sizeof(void*), false );
        hostOrDeviceCpy(((char*)m_leafQueue->ptr<Store<u32>>()) + offsetof(Store<u32>, m_max), &m_numRayLeafQueries, sizeof(u32), true);
        resetRayLeafQueue();
        for ( u32 i=0; i<2; i++ )
        {
            u64 pRayBuffer = (u64)m_rayBoxBuffer[i]->ptr<void>();
            hostOrDeviceCpy(((char*)m_rayBoxQueue[i]->ptr<Store<u32>>()) + offsetof(Store<u32>, m_elements), &pRayBuffer, sizeof(void*), false);
            hostOrDeviceCpy(((char*)m_rayBoxQueue[i]->ptr<Store<u32>>()) + offsetof(Store<u32>, m_max), &m_numRayBoxQueries, sizeof(u32), true);
            resetRayBoxQueue(i);
        }
        
    }

    Tracer::~Tracer()
    {
        delete m_leafQueue;
        delete m_leafBuffer;
        delete m_rayBoxQueue[0];
        delete m_rayBoxQueue[1];
        delete m_rayBoxBuffer[0];
        delete m_rayBoxBuffer[1];

    }

    void Tracer::resetRayBoxQueue(u32 idx)
    {
        u32 zero=0;
        hostOrDeviceCpy(((char*)m_rayBoxQueue[idx]->ptr<Store<u32>>()) + offsetof(Store<u32>, m_top), &zero, sizeof(u32), true);
        
    }

    void Tracer::resetRayLeafQueue()
    {
        u32 zero=0;
        hostOrDeviceCpy(((char*)m_leafQueue->ptr<Store<u32>>()) + offsetof(Store<u32>, m_top), &zero, sizeof(u32), true);
        
    }

    u32 Tracer::trace(const float* eye3, const float* orient3x3, const IGpuStaticScene* scene, const ITraceQuery* query, const ITraceResult* const* results, u32 numResults)
    {
        if ( !scene || !query || numResults==0 || results==nullptr )
        {
            return ERROR_INVALID_PARAMETER;
        }

        for ( u32 i=0; i<numResults; ++i )
            if ( !results[i] ) return ERROR_INVALID_PARAMETER;

        const u32 numRays = 256*256; // A single tile

        Store<RayBox>* rbQueue[]    = { m_rayBoxQueue[0]->ptr<Store<RayBox>>(), m_rayBoxQueue[1]->ptr<Store<RayBox>>() };
        Store<u32>* leafQueue       = m_leafQueue->ptr<Store<u32>>();
        const vec3* rayDirs         = static_cast<const TraceQuery*>( query )->m_dirs->ptr<const vec3>();
        const BvhNode* bvhNodes     = static_cast<const GpuStaticScene*>( scene )->m_bvhTree->ptr<const BvhNode>();

        for ( u32 i=0; i<1; i++ )
        {
            resetRayBoxQueue((i+1)%2);
            u32 err = rlRayBox(eye3, orient3x3, numRays, rbQueue[i%2], rbQueue[(i+1)%2], leafQueue, rayDirs, bvhNodes);
            assert(err==0);
        }

        return ERROR_ALL_FINE;
    }

}
