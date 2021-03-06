#include "Tracer.h"
#include "GpuStaticScene.h"
#include "DeviceBuffer.h"
#include "Reylax_internal.h"
using namespace std;


namespace Reylax
{
    GLOBAL_DYN void TileKernel(u32 numRays, u32 tileOffset);
    void UpdateTraceContext(const TracerContext& ct, bool wait);
    void UpdateTraceData(const vec3& eye, mat3& orient, vec3* rays, u32* pixels);

    Profiler CpuProfiler;

    ITracer* ITracer::create(u32 numRaysPerTile, u32 maxRecursionDepth)
    {
        return new Tracer( numRaysPerTile, maxRecursionDepth );
    }

    Tracer::Tracer(u32 numRaysPerTile, u32 maxRecursionDepth):
        m_numRaysPerTile(numRaysPerTile)
    {
        m_ctx.maxDepth = maxRecursionDepth;

    #if RL_PRINT_STATS
        printf("\n--- Tracer allocations ---\n\n");
    #endif

        u32 numQueries = m_numRaysPerTile;// TODO check allocation style, multiple queries per ray possible, so this is not correct *maxRecursionDepth;
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
        u32 partMax = numQueries/RL_NUMMER_INNER_QUEUES;
        assert( numQueries % RL_NUMMER_INNER_QUEUES == 0 );
        for ( u32 i=0; i<2; i++ )
        {
            COPY_PTR_TO_DEVICE_ASYNC(m_pointBoxQueue[i], m_pointBoxBuffer[i], Store<PointBox>, m_elements);
            COPY_VALUE_TO_DEVICE_ASYNC(m_pointBoxQueue[i], numQueries, Store<PointBox>, m_max, sizeof(u32));
            COPY_VALUE_TO_DEVICE_ASYNC(m_pointBoxQueue[i], partMax, Store<PointBox>, m_partMax, sizeof(u32));
            COPY_PTR_TO_DEVICE_ASYNC(m_leafQueue[i], m_leafBuffer[i], Store<RayLeaf>, m_elements);
            COPY_VALUE_TO_DEVICE_ASYNC(m_leafQueue[i], numQueries, Store<RayLeaf>, m_max, sizeof(u32));
            COPY_VALUE_TO_DEVICE_ASYNC(m_leafQueue[i], partMax, Store<PointBox>, m_partMax, sizeof(u32));
        }
        COPY_PTR_TO_DEVICE_ASYNC(m_rayQueue, m_rayBuffer, Store<Ray>, m_elements);
        COPY_VALUE_TO_DEVICE_ASYNC(m_rayQueue, numQueries, Store<Ray>, m_max, sizeof(u32));
        COPY_VALUE_TO_DEVICE_ASYNC(m_rayQueue, partMax, Store<Ray>, m_partMax, sizeof(u32));

        m_id2Queue = new DeviceBuffer(sizeof(char)*m_numRaysPerTile);
      //  m_id2RayQueue = new DeviceBuffer(sizeof(char)*m_numRaysPerTile);

        m_ctx.rayPayload    = m_rayQueue->ptr<Store<Ray>>();
        m_ctx.pbQueueIn     = m_pointBoxQueue[0]->ptr<Store<PointBox>>();
        m_ctx.pbQueueOut    = m_pointBoxQueue[1]->ptr<Store<PointBox>>();
        m_ctx.leafQueueIn   = m_leafQueue[0]->ptr<Store<RayLeaf>>();
        m_ctx.leafQueueOut  = m_leafQueue[1]->ptr<Store<RayLeaf>>();
        m_ctx.hitResults    = m_hitResults->ptr<HitResult>();
        m_ctx.id2Queue      = m_id2Queue->ptr<byte>();
     //   m_ctx.id2RayQueue   = m_id2RayQueue->ptr<char>();

        u64 totalMemory = 0;
        for ( u32 i=0; i<2; i++ )
        {
            totalMemory += m_pointBoxBuffer[i]->size() + m_pointBoxQueue[i]->size();
            totalMemory += m_leafBuffer[i]->size() + m_leafQueue[i]->size();
            totalMemory += m_id2Queue->size();// + m_id2RayQueue->size();
        }
        totalMemory += m_rayQueue->size() + m_rayBuffer->size();

        printf("Total: %.3fmb\n", (float)totalMemory/1024/1024);
        printf("\n--- End Tracer allocations ---\n\n");

        // Ensure all memory transfers are done
        SyncDevice();
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
        delete m_id2Queue;
      //  delete m_id2RayQueue;
    }

    u32 Tracer::trace(u32 numRays, const IGpuStaticScene* scene, RaySetupFptr setupCb, HitResultFptr hitCb)
    {
        if ( !scene /*|| !setupCb || !hitCb*/ )
        {
            return ERROR_INVALID_PARAMETER;
        }

        GpuStaticScene* scn = (GpuStaticScene*)(scene);

        if ( scn != m_lastTracedScene )
        {
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
            UpdateTraceContext(m_ctx, false);
            m_lastTracedScene = scn;
        }

        CpuProfiler.beginProfile();

        // while rays to process, process per batch/tile to conserve memory usage
        u32 kTile=0;
        u32 tileOffset=0;
        while ( tileOffset != numRays )
        {
            u32 numRaysTile = min( m_numRaysPerTile, numRays-tileOffset );

            CpuProfiler.start();
            RL_KERNEL_CALL_DYN( 1, 1, 1, TileKernel, numRaysTile, tileOffset );
            CpuProfiler.stop( "Tile " + to_string(kTile) );

            kTile++;
            tileOffset += numRaysTile;
        }

        CpuProfiler.endProfile("Trace");
  
        return ERROR_ALL_FINE;
    }

    u32 Tracer::trace2(u32 numRays, const IGpuStaticScene* scene, const float* eye3, const float* orient3x3, const float* rays3, const u32* pixels)
    {
        UpdateTraceData( (vec3&)*eye3, (mat3&)*orient3x3, (vec3*)rays3, (u32*)pixels );
        return trace( numRays, scene, nullptr, nullptr );
    }

    QueueRayFptr Tracer::getQueueRayAddress() const
    {
        return nullptr;
    }
}