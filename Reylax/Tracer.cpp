#include "Tracer.h"
#include "SmallGpuStructs.h"
#include "DeviceBuffer.h"
#include "Reylax_internal.h"
using namespace std;


namespace Reylax
{
    GLOBAL void TileKernel(u32 numRays);
    DEVICE void QueueRay(u32 localId, const float* ori, const float* dir);

    extern DEVICE TracerContext ct;


    ITracer* ITracer::create(u32 numRaysPerTile, u32 maxRecursionDepth)
    {
        u32 queueLength = numRaysPerTile*maxRecursionDepth;
        return new Tracer( queueLength, queueLength, queueLength, numRaysPerTile);
    }

    Tracer::Tracer(u32 numPointBoxQueries, u32 numLeafQueries, u32 numRayQueries, u32 numRaysPerTile):
        m_numPointBoxQueries(numPointBoxQueries),
        m_numRayLeafQueries(numLeafQueries),
        m_numRayQueries(numRayQueries),
        m_numRaysPerTile(numRaysPerTile),
        m_rayQueue(nullptr),
        m_rayBuffer(nullptr)
    {
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
        
        for ( u32 i=0; i<2; i++ )
        {
            m_pointBoxQueue[i]  = new DeviceBuffer(sizeof(Store<PointBox>));
            m_pointBoxBuffer[i] = new DeviceBuffer(numPointBoxQueries*sizeof(PointBox));
            m_leafQueue[i]  = new DeviceBuffer(sizeof(Store<RayLeaf>));
            m_leafBuffer[i] = new DeviceBuffer(numLeafQueries*sizeof(RayLeaf));
        #if RL_PRINT_STATS
            printf("PointBoxQueries %d, count %d, size %.3fmb\n", i, numPointBoxQueries, (float)m_pointBoxBuffer[i]->size()/1024/1024);
            printf("RayLeafQueries %d, count %d, size %.3fmb\n", i, numLeafQueries, (float)m_leafBuffer[i]->size()/1024/1024);
        #endif
        }
        m_rayQueue  = new DeviceBuffer(sizeof(Store<Ray>));
        m_rayBuffer = new DeviceBuffer(numRayQueries*sizeof(Ray));
    #if RL_PRINT_STATS
        printf("RayQueries, count %d, size %.3fmb\n", numRayQueries, (float)m_rayBuffer->size()/1024/1024);
    #endif

        // Assign device buffers to queue elements ptr
        for ( u32 i=0; i<2; i++ )
        {
            COPY_PTR_TO_DEVICE_ASYNC(m_pointBoxQueue[i], m_pointBoxBuffer[i], Store<PointBox>, m_elements);
            COPY_VALUE_TO_DEVICE_ASYNC(m_pointBoxQueue[i], numPointBoxQueries, Store<PointBox>, m_max, sizeof(u32));
            COPY_PTR_TO_DEVICE_ASYNC(m_leafQueue[i], m_leafBuffer[i], Store<RayLeaf>, m_elements);
            COPY_VALUE_TO_DEVICE_ASYNC(m_leafQueue[i], numLeafQueries, Store<RayLeaf>, m_max, sizeof(u32));
        }
        COPY_PTR_TO_DEVICE_ASYNC(m_rayQueue, m_rayBuffer, Store<Ray>, m_elements);
        COPY_VALUE_TO_DEVICE_ASYNC(m_rayQueue, numRayQueries, Store<Ray>, m_max, sizeof(u32));

        RL_CUDA_CALL( cudaMemcpyToSymbolAsync( &ct.rayPayload, m_rayQueue->ptr<void>(), sizeof(void*), 0 ) );
        RL_CUDA_CALL( cudaMemcpyToSymbolAsync( &ct.pbQueues[0], m_pointBoxBuffer[0]->ptr<void>(), sizeof(void*), 0 ) );
        RL_CUDA_CALL( cudaMemcpyToSymbolAsync( &ct.pbQueues[1], m_pointBoxBuffer[1]->ptr<void>(), sizeof(void*), 0 ) );
        RL_CUDA_CALL( cudaMemcpyToSymbolAsync( &ct.leafQueues[0], m_leafQueue[0]->ptr<void>(), sizeof(void*), 0 ) );
        RL_CUDA_CALL( cudaMemcpyToSymbolAsync( &ct.leafQueues[1], m_leafQueue[1]->ptr<void>(), sizeof(void*), 0 ) );

        u64 totalMemory = 0;
        for ( u32 i=0; i<2; i++ )
        {
            totalMemory += m_pointBoxBuffer[i]->size() + m_pointBoxQueue[i]->size();
            totalMemory += m_leafBuffer[i]->size() + m_leafQueue[i]->size();
        }
        
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
        MeshData** meshData             = scn->m_meshDataPtrs->ptr<MeshData*>();

        RL_CUDA_CALL( cudaMemcpyToSymbolAsync( &ct.bMin, &scn->bMin, sizeof(vec3) ) );
        RL_CUDA_CALL( cudaMemcpyToSymbolAsync( &ct.bMax, &scn->bMax, sizeof(vec3) ) );
        RL_CUDA_CALL( cudaMemcpyToSymbolAsync( &ct.bvhNodes, bvhNodes, sizeof(BvhNode*) ) );
        RL_CUDA_CALL( cudaMemcpyToSymbolAsync( &ct.faces, faces, sizeof(Face*) ) );
        RL_CUDA_CALL( cudaMemcpyToSymbolAsync( &ct.faceClusters, faceClusters, sizeof(FaceCluster*) ) );
        RL_CUDA_CALL( cudaMemcpyToSymbolAsync( &ct.sides, sides, sizeof(u32*) ) );
        RL_CUDA_CALL( cudaMemcpyToSymbolAsync( &ct.meshData, meshData, sizeof(MeshData**) ) );
        RL_CUDA_CALL( cudaMemcpyToSymbolAsync( &ct.setupCb, setupCb, sizeof(RaySetupFptr) ) );
        RL_CUDA_CALL( cudaMemcpyToSymbolAsync( &ct.hitCb, hitCb, sizeof(HitResultFptr) ) );

        m_profiler.beginProfile();

        // while rays to process, process per batch/tile to conserve memory usage
        u32 kTile=0;
        while ( numRays > 0 )
        {
            u32 numRaysThisTile = numRays;
            if ( m_numRaysPerTile < numRaysThisTile ) numRaysThisTile = m_numRaysPerTile;
            m_profiler.start();

            RL_KERNEL_CALL(1, 1, 1, TileKernel, numRaysThisTile);

            m_profiler.stop("Tile " + to_string(kTile++));
            numRays -= numRaysThisTile;
        }
        m_profiler.endProfile("Trace");
  
        return ERROR_ALL_FINE;
    }

    QueueRayFptr Tracer::getQueueRayAddress() const
    {
        return QueueRay;
    }
}