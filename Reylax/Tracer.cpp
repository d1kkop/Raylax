#include "Tracer.h"
#include "SmallGpuStructs.h"
#include "DeviceBuffer.h"
#include "Reylax_internal.h"
using namespace std;


namespace Reylax
{
    GLOBAL void TileKernel(u32 numRays,
                           vec3 eye,
                           mat3 orient,
                           Store<RayBox>** rbQueues,
                           Store<RayBox>* leafQueue,
                           Store<RayFace>* rayFaceQueue,
                           const vec3* rayOris,
                           const vec3* rayDirs,
                           const BvhNode* bvhNodes,
                           const Face* faces,
                           const FaceCluster* faceClusters,
                           HitCluster* hitResultClusters,
                           const MeshData* const* meshData,
                           HitResult** hitResults, u32 numResults);


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
        m_rayFaceBuffer(nullptr),
        m_hitResultClusters(nullptr)
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
        m_hitResultClusters = new DeviceBuffer(numRaysPerTile*sizeof(HitCluster));

    #if RL_PRINT_STATS
        printf("\n--- Tracer allocations ---\n\n");
        printf("RayLeafQueries count %d, size %.3fmb\n", numLeafQueries, (float)m_leafBuffer->size()/1024/1024);
        printf("RayFaceQueries count %d, size %.3fmb\n", numRayFaceQueries, (float)m_rayFaceBuffer->size()/1024/1024);
        printf("HitClusters count %d, size %.3fmb\n", numRaysPerTile, (float)m_hitResultClusters->size()/1024/1024);
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
        COPY_PTR_TO_DEVICE_ASYNC( m_leafQueue, m_leafBuffer, Store<RayBox>, m_elements );
        COPY_PTR_TO_DEVICE_ASYNC( m_rayFaceQueue, m_rayFaceBuffer, Store<RayFace>, m_elements);
        COPY_VALUE_TO_DEVICE_ASYNC( m_leafQueue, m_numRayLeafQueries, Store<RayBox>, m_max, sizeof(u32) );
        COPY_VALUE_TO_DEVICE_ASYNC( m_rayFaceQueue, numRayFaceQueries, Store<RayFace>, m_max, sizeof(u32));
        for ( u32 i=0; i<2; i++ )
        {
            u64 pRayBuffer = (u64)m_rayBoxBuffer[i]->ptr<void>();
            COPY_PTR_TO_DEVICE_ASYNC(m_rayBoxQueue[i], m_rayBoxBuffer[i], Store<RayBox>, m_elements);
            COPY_VALUE_TO_DEVICE_ASYNC(m_rayBoxQueue[i], m_numRayBoxQueries, Store<RayBox>, m_max, sizeof(u32));
        }

        u64 totalMemory = 0;
        totalMemory += m_leafBuffer->size() + m_rayFaceBuffer->size() + m_leafQueue->size() + m_rayFaceQueue->size() + m_hitResultClusters->size();
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
        delete m_hitResultClusters;
    }

    u32 Tracer::trace(const float* eye3, const float* orient3x3,
                      const IGpuStaticScene* scene, const ITraceQuery* query, 
                      const ITraceResult* const* results, u32 numResults)
    {
        if ( !eye3 || !orient3x3 || !scene || !query || numResults==0 || results==nullptr || numResults > RL_RAY_ITERATIONS )
        {
            return ERROR_INVALID_PARAMETER;
        }

        auto trq = static_cast<const TraceQuery*>(query);
        auto scn = static_cast<const GpuStaticScene*>(scene);

        HitResult* hitResults[RL_RAY_ITERATIONS];
        for ( u32 i=0; i<numResults; ++i )
        {
            if ( !results[i] ) return ERROR_INVALID_PARAMETER;
            hitResults[i] = static_cast<const TraceResult*>( results[i] )->m_result->ptr<HitResult>();
        }

        vec3 eye                        = *(vec3*)eye3;
        mat3 orient                     = *(mat3*)orient3x3;
        Store<RayBox>* rbQueues[]       = { m_rayBoxQueue[0]->ptr<Store<RayBox>>(), m_rayBoxQueue[1]->ptr<Store<RayBox>>() };
        Store<RayBox>* leafQueue        = m_leafQueue->ptr<Store<RayBox>>();
        Store<RayFace>* rayFaceQueue    = m_rayFaceQueue->ptr<Store<RayFace>>();
        const vec3* rayOris             = trq->m_oris->ptr<const vec3>();
        const vec3* rayDirs             = trq->m_dirs->ptr<const vec3>();
        const BvhNode* bvhNodes         = scn->m_bvhTree->ptr<const BvhNode>();
        const Face* faces               = scn->m_faces->ptr<const Face>();
        const FaceCluster* faceClusters = scn->m_faceClusters->ptr<const FaceCluster>();
        HitCluster* hitclusters         = m_hitResultClusters->ptr<HitCluster>();
        MeshData** meshData             = scn->m_meshDataPtrs->ptr<MeshData*>();

        u32 totalRays = trq->m_numRays;
        m_profiler.beginProfile();

        // while rays to process, process per batch/tile to conserve memory usage
        u32 kTile=0;
        while ( totalRays > 0 )
        {
            u32 numRaysThisTile = _min(m_numRaysPerTile, totalRays);
            m_profiler.start();

            RL_KERNEL_CALL(1, 1, 1, TileKernel,
                           numRaysThisTile, eye, orient,
                          (Store<RayBox>**) rbQueues, leafQueue, rayFaceQueue,
                           rayOris, rayDirs,
                           bvhNodes, faces, faceClusters,
                           hitclusters, meshData,
                           (HitResult**)hitResults, numResults);

            m_profiler.stop("Tile " + to_string(kTile++));
            totalRays -= numRaysThisTile;
        }

        m_profiler.endProfile("Trace");

        return ERROR_ALL_FINE;
    }

}