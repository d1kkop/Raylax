#include "Tracer.h"
#include "GpuStaticScene.h"
#include "DeviceBuffer.h"
#include "Reylax_internal.h"
using namespace std;


namespace Reylax
{
    GLOBAL void TileKernel(u32 numRays, u32 tileOffset);
    DEVICE void QueueRay(const float* ori, const float* dir);
    extern void UpdateTraceContext(const TracerContext& ct, bool wait);


    ITracer* ITracer::create(u32 numRaysPerTile, u32 maxRecursionDepth)
    {
        return new Tracer( numRaysPerTile, maxRecursionDepth );
    }

    Tracer::Tracer(u32 numRaysPerTile, u32 maxRecursionDepth):
        m_numRaysPerTile(numRaysPerTile),
        m_rayQueue(nullptr),
        m_rayBuffer(nullptr),
        m_hitResults(nullptr)
    {
        memset(&m_ctx, 0, sizeof(TracerContext));
        m_ctx.maxDepth = maxRecursionDepth;

        for ( u32 i=0; i<2; i++ )
        {
            m_pointBoxQueue[i]  = nullptr;
            m_pointBoxBuffer[i] = nullptr;
            m_leafQueue[i]  = nullptr;
            m_leafBuffer[i] = nullptr;
        }

    #if RL_PRINT_STATS
        printf("\n--- Tracer allocations ---\n\n");
    #endif

        u32 numQueries = m_numRaysPerTile*maxRecursionDepth;
        for ( u32 i=0; i<2; i++ )
        {
            m_pointBoxQueue[i]  = new DeviceBuffer(sizeof(Store<PointBox>));
            m_pointBoxBuffer[i] = new DeviceBuffer(numQueries*sizeof(PointBox));
            m_leafQueue[i]  = new DeviceBuffer(sizeof(Store<RayLeaf>));
            m_leafBuffer[i] = new DeviceBuffer(numQueries*sizeof(RayLeaf));
        #if RL_PRINT_STATS
            printf("PointBoxQueries %.3fmb\n", (float)m_pointBoxBuffer[i]->size()/1024/1024);
            printf("RayLeafQueries %.3fmb\n", (float)m_leafBuffer[i]->size()/1024/1024);
        #endif
        }
        m_rayQueue   = new DeviceBuffer(sizeof(Store<Ray>));
        m_rayBuffer  = new DeviceBuffer(numQueries*sizeof(Ray));
        m_hitResults = new DeviceBuffer(numRaysPerTile*sizeof(HitResult));
    #if RL_PRINT_STATS
        printf("RayQueries %.3fmb\n", (float)m_rayBuffer->size()/1024/1024);
        printf("HitResults %.3fmb\n", (float)m_hitResults->size()/1024/1024);
    #endif

        // Assign device buffers to queue elements ptr
        u32 zero=0;
        for ( u32 i=0; i<2; i++ )
        {
            COPY_PTR_TO_DEVICE_ASYNC(m_pointBoxQueue[i], m_pointBoxBuffer[i], Store<PointBox>, m_elements);
            COPY_VALUE_TO_DEVICE_ASYNC(m_pointBoxQueue[i], numQueries, Store<PointBox>, m_max, sizeof(u32));
            COPY_VALUE_TO_DEVICE_ASYNC(m_pointBoxQueue[i], zero, Store<PointBox>, m_top, sizeof(u32));
            COPY_PTR_TO_DEVICE_ASYNC(m_leafQueue[i], m_leafBuffer[i], Store<RayLeaf>, m_elements);
            COPY_VALUE_TO_DEVICE_ASYNC(m_leafQueue[i], numQueries, Store<RayLeaf>, m_max, sizeof(u32));
            COPY_VALUE_TO_DEVICE_ASYNC(m_leafQueue[i], zero, Store<RayLeaf>, m_top, sizeof(u32));
        }
        COPY_PTR_TO_DEVICE_ASYNC(m_rayQueue, m_rayBuffer, Store<Ray>, m_elements);
        COPY_VALUE_TO_DEVICE_ASYNC(m_rayQueue, numQueries, Store<Ray>, m_max, sizeof(u32));
        COPY_VALUE_TO_DEVICE_ASYNC(m_rayQueue, zero, Store<Ray>, m_top, sizeof(u32));

        m_ctx.rayPayload    = m_rayQueue->ptr<Store<Ray>>();
        m_ctx.pbQueues[0]   = m_pointBoxQueue[0]->ptr<Store<PointBox>>();
        m_ctx.pbQueues[1]   = m_pointBoxQueue[1]->ptr<Store<PointBox>>();
        m_ctx.leafQueues[0] = m_leafQueue[0]->ptr<Store<RayLeaf>>();
        m_ctx.leafQueues[1] = m_leafQueue[1]->ptr<Store<RayLeaf>>();
        m_ctx.hitResults    = m_hitResults->ptr<HitResult>();

        u64 totalMemory = 0;
        for ( u32 i=0; i<2; i++ )
        {
            totalMemory += m_pointBoxBuffer[i]->size() + m_pointBoxQueue[i]->size();
            totalMemory += m_leafBuffer[i]->size() + m_leafQueue[i]->size();
        }
        totalMemory += m_rayQueue->size() + m_rayBuffer->size();

        printf("Total: %.3fmb\n", (float)totalMemory/1024/1024);
        printf("\n--- End Tracer allocations ---\n\n");

        // Ensure all memory transfers are done
        syncDevice();
    }

    Tracer::~Tracer()
    {
        for ( u32 i=0; i<2; ++i )
        {
            delete m_leafQueue[i];
            delete m_leafBuffer[i];
            delete m_pointBoxQueue[i];
            delete m_pointBoxBuffer[i];
        }
        delete m_rayQueue;
        delete m_rayBuffer;
        delete m_hitResults;
    }

    u32 Tracer::trace(u32 numRays, const IGpuStaticScene* scene, RaySetupFptr setupCb, HitResultFptr hitCb)
    {
        if ( !scene || !setupCb || !hitCb )
        {
            return ERROR_INVALID_PARAMETER;
        }

        auto scn = static_cast<const GpuStaticScene*>(scene);

        const BvhNode* bvhNodes         = scn->m_bvhTree->ptr<const BvhNode>();
        const Face* faces               = scn->m_faces->ptr<const Face>();
        const FaceCluster* faceClusters = scn->m_faceClusters->ptr<const FaceCluster>();
        const u32* sides                = scn->m_sides->ptr<const u32>();
        MeshData** meshDataPtrs         = scn->m_meshDataPtrs->ptr<MeshData*>();

        m_ctx.setupCb               = setupCb;
        m_ctx.hitCb                 = hitCb;
        m_ctx.bMin                  = scn->bMin;
        m_ctx.bMax                  = scn->bMax;
        m_ctx.bvhNodes              = bvhNodes;
        m_ctx.faces                 = faces;
        m_ctx.faceClusters          = faceClusters;
        m_ctx.sides                 = sides;
        m_ctx.meshDataPtrs          = meshDataPtrs;
        UpdateTraceContext( m_ctx, false );

        m_profiler.beginProfile();

        // while rays to process, process per batch/tile to conserve memory usage
        u32 kTile=0;
        u32 tileOffset=0;
        while ( tileOffset != numRays )
        {
            u32 numRaysTile = min( m_numRaysPerTile, numRays-tileOffset );

            m_profiler.start();
            RL_KERNEL_CALL( 1, 1, 1, TileKernel, numRaysTile, tileOffset );
            m_profiler.stop( "Tile " + to_string(kTile) );

            kTile++;
            tileOffset += numRaysTile;
        }
        m_profiler.endProfile("Trace");
  
        return ERROR_ALL_FINE;
    }

    QueueRayFptr Tracer::getQueueRayAddress() const
    {
        return QueueRay;
    }
}