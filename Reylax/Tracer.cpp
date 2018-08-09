#include "Tracer.h"
#include "SmallGpuStructs.h"
#include "DeviceBuffer.h"
#include "Reylax_internal.h"
using namespace std;

extern "C"
{
    using namespace Reylax;
    u32 rlResetRayBoxQueue(Store<RayBox>* queue);
    u32 rlResetRayLeafQueue(Store<RayBox>* queue);
    u32 rlResetRayFaceQueue(Store<RayFace>* queue);
    u32 rlSetInitialRbQueries(u32 numRays, Store<RayBox>* queue);
    u32 rlRayBox(const float* eye3, 
                 const float* orient3x3, 
                 u32 numRayBoxes,
                 Store<RayBox>* rayBoxQueueIn, 
                 Store<RayBox>* rayBoxQueueOut, 
                 Store<RayBox>* leafQueue,
                 const vec3* rayDirs,
                 const BvhNode* bvhNodes);
    u32 rlExpandLeafs(u32 numRayLeafQueries,
                      Store<RayBox>* leafQueue,
                      Store<RayFace>* rayFaceQueue,
                      const BvhNode* bvhNodes,
                      const FaceCluster* faceClusters);
}


namespace Reylax
{
    ITracer* ITracer::create()
    {
        return new Tracer();
    }

    Tracer::Tracer(u32 numRayBoxQueries, u32 numLeafQueries, u32 numRayFaceQueries, u32 numRaysPerTile):
        m_numRayBoxQueries(numRayBoxQueries),
        m_numRayLeafQueries(numLeafQueries),
        m_numRaysPerTile(numRaysPerTile),
        m_leafQueue(nullptr),
        m_leafBuffer(nullptr),
        m_rayFaceQueue(nullptr),
        m_rayFaceBuffer(nullptr)
    {
        for ( u32 i=0; i<2; i++ )
        {
            m_rayBoxQueue[i]  = nullptr;
            m_rayBoxBuffer[i] = nullptr;
        }
        m_leafQueue  = new DeviceBuffer(sizeof(Store<RayBox>));
        m_leafBuffer = new DeviceBuffer(numLeafQueries*sizeof(RayBox));
        m_rayFaceQueue  = new DeviceBuffer(sizeof(Store<RayFace>));
        m_rayFaceBuffer = new DeviceBuffer(numRayFaceQueries*sizeof(RayFace));

    #if RL_PRINT_STATS
        printf("--- Tracer allocations ---\n\n");
        printf("RayLeafQueries count %d, size %.3fmb\n", numLeafQueries, (float)m_leafBuffer->size()/1024/1024);
        printf("RayFaceQueries count %d, size %.3fmb\n", numRayFaceQueries, (float)m_rayFaceBuffer->size()/1024/1024);
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
        u64 pLeafBuffer     = (u64)m_leafBuffer->ptr<void>();
        u64 pRayFaceBuffer  = (u64)m_rayFaceBuffer->ptr<void>();
        hostOrDeviceCpy(m_leafQueue->ptr<char>() + offsetof(Store<u32>, m_elements), &pLeafBuffer, sizeof(void*), cudaMemcpyHostToDevice, false);
        hostOrDeviceCpy(m_leafQueue->ptr<char>() + offsetof(Store<u32>, m_max), &m_numRayLeafQueries, sizeof(u32), cudaMemcpyHostToDevice, false);
        hostOrDeviceCpy(m_rayFaceQueue->ptr<char>() + offsetof(Store<u32>, m_elements), &pRayFaceBuffer, sizeof(void*), cudaMemcpyHostToDevice, false);
        hostOrDeviceCpy(m_rayFaceQueue->ptr<char>() + offsetof(Store<u32>, m_max), &m_numRayFaceQueries, sizeof(u32), cudaMemcpyHostToDevice, false);
        for ( u32 i=0; i<2; i++ )
        {
            u64 pRayBuffer = (u64)m_rayBoxBuffer[i]->ptr<void>();
            hostOrDeviceCpy(m_rayBoxQueue[i]->ptr<char>() + offsetof(Store<u32>, m_elements), &pRayBuffer, sizeof(void*), cudaMemcpyHostToDevice, false);
            hostOrDeviceCpy(m_rayBoxQueue[i]->ptr<char>() + offsetof(Store<u32>, m_max), &m_numRayBoxQueries, sizeof(u32), cudaMemcpyHostToDevice, false);
        }

        u64 totalMemory = 0;
        totalMemory += m_leafBuffer->size() + m_rayFaceBuffer->size() + m_leafQueue->size() + m_rayFaceQueue->size();
        for ( u32 i=0; i<2; i++) totalMemory += m_rayBoxBuffer[i]->size() + m_rayBoxQueue[i]->size();
        
        printf("Total: %.3fmb\n", (float)totalMemory/1024/1024);
        printf("\n--- End Tracer allocations ---\n\n");
    }

    Tracer::~Tracer()
    {
        delete m_leafQueue;
        delete m_leafBuffer;
        delete m_rayFaceQueue;
        delete m_rayFaceBuffer;
        delete m_rayBoxQueue[0];
        delete m_rayBoxQueue[1];
        delete m_rayBoxBuffer[0];
        delete m_rayBoxBuffer[1];
    }

    u32 Tracer::getQueueTop(const char* queue, bool wait)
    {
        u32 top=0;
        u32 err = hostOrDeviceCpy(&top, queue + offsetof(Store<RayBox>, m_top), sizeof(u32), cudaMemcpyDeviceToHost, !wait);
        assert(err==0);
        return top;
    }

    u32 Tracer::trace(const float* eye3, const float* orient3x3, const IGpuStaticScene* scene, const ITraceQuery* query, const ITraceResult* const* results, u32 numResults)
    {
        if ( !scene || !query || numResults==0 || results==nullptr )
        {
            return ERROR_INVALID_PARAMETER;
        }

        for ( u32 i=0; i<numResults; ++i )
            if ( !results[i] ) return ERROR_INVALID_PARAMETER;

        auto trq = static_cast<const TraceQuery*>(query);
        auto scn = static_cast<const GpuStaticScene*>(scene);

        Store<RayBox>* rbQueue[]        = { m_rayBoxQueue[0]->ptr<Store<RayBox>>(), m_rayBoxQueue[1]->ptr<Store<RayBox>>() };
        Store<RayBox>* leafQueue        = m_leafQueue->ptr<Store<RayBox>>();
        Store<RayFace>* rayFaceQueue    = m_rayFaceQueue->ptr<Store<RayFace>>();
        const vec3* rayDirs             = trq->m_dirs->ptr<const vec3>();
        const BvhNode* bvhNodes         = scn->m_bvhTree->ptr<const BvhNode>();
        const Face* faces               = scn->m_faces->ptr<const Face>();
        const FaceCluster* faceClusters = scn->m_faceClusters->ptr<const FaceCluster>();

        u32 totalRays = trq->m_numRays;
        m_profiler.beginProfile();

        // while rays to process, process per batch/tile to conserve memory usage
        while ( totalRays > 0 )
        {
            u32 numRaysThisTile = _min(m_numRaysPerTile, totalRays);

        //#if RL_PRINT_STATS
        //    printf("--- Trace Tile begin ---\n\n");
        //#endif

            // prepare tile
            rlResetRayBoxQueue(rbQueue[0]);
            rlResetRayLeafQueue(leafQueue);
            rlResetRayFaceQueue(rayFaceQueue);
            rlSetInitialRbQueries(numRaysThisTile, rbQueue[0]);

            // execute ray/box queries
            m_profiler.start();
            doRayBoxQueries(rbQueue, leafQueue, eye3, orient3x3, rayDirs, bvhNodes);
            m_profiler.stop("Rb queries");

            // execute ray/leaf queries
            m_profiler.start();
            doExpandLeafs(leafQueue, rayFaceQueue, bvhNodes, faceClusters);
            m_profiler.stop("Lf queries");

            // execute ray/face queries
            u32 rfTop = getQueueTop( (const char*)rayFaceQueue, true );

            // go next tile
            totalRays -= numRaysThisTile; 

        //#if RL_PRINT_STATS
        //    printf("\n--- Trace Tile end ---\n\n");
        //#endif
        }

        m_profiler.endProfile("Trace");

        return ERROR_ALL_FINE;
    }

    void Tracer::doRayBoxQueries(Store<RayBox>** rbQueue, Store<RayBox>* leafQueue, const float* eye3, const float* orient3x3, const vec3* rayDirs, const BvhNode* bvhNodes)
    {
        for ( u32 i=0; /*no stop cond*/; ++i )
        {
            u32 inQueue  = i%2;
            u32 outQueue = (i+1)%2;

            // stop if all rb queries are done
            u32 topIn = getQueueTop((const char*)rbQueue[inQueue], true);
            if ( topIn==0 )
            {
                break;
            }

            // ensure top out is set to zero
            rlResetRayBoxQueue(rbQueue[outQueue]);

        #if !RL_CUDA && DBG_RB_QUERIES
            printf("--- RB iteration %d ---\n", i);
            printf("Num RB In/Out Before %d/%d\n", rbQueue[inQueue]->m_top, rbQueue[outQueue]->m_top);
            printf("Num LeafQ %d Before\n", leafQueue->m_top);
        #endif

            // execute all rb queries from queue-in and generate new to queue-out or leaf-queue
            u32 err = rlRayBox(eye3, orient3x3, topIn, rbQueue[inQueue], rbQueue[outQueue], leafQueue, rayDirs, bvhNodes);
            assert(err==0);

        #if !RL_CUDA && DBG_RB_QUERIES
            printf("Num RB In/Out After %d/%d\n", rbQueue[inQueue]->m_top, rbQueue[outQueue]->m_top);
            printf("Num LeafQ %d After\n", leafQueue->m_top);
        #endif
        }
    }

    void Tracer::doExpandLeafs(Store<RayBox>* leafQueue, Store<RayFace>* rayFaceQueue, const BvhNode* bvhNodes, const FaceCluster* faceClusters)
    {
    #if !RL_CUDA && DBG_RL_QUERIES
        printf("Num in LeafQ %d\n", leafQueue->m_top);
    #endif
        u32 numLeafQueries = getQueueTop((const char*)leafQueue, true);
        u32 err = rlExpandLeafs(numLeafQueries, leafQueue, rayFaceQueue, bvhNodes, faceClusters);
        assert(err==0);
    }

}