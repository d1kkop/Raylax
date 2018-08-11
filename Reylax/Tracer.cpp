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
                           Store<RayLeaf>** leafQueues,
                           vec3* rayOris,
                           const vec3* rayDirs,
                           const BvhNode* bvhNodes,
                           const Face* faces,
                           const FaceCluster* faceClusters,
                           const MeshData* const* meshData,
                           HitResult* hitResults);


    ITracer* ITracer::create()
    {
        return new Tracer();
    }

    Tracer::Tracer(u32 numRayBoxQueries, u32 numLeafQueries, u32 numRaysPerTile):
        m_numRayBoxQueries(numRayBoxQueries),
        m_numRayLeafQueries(numLeafQueries),
        m_numRaysPerTile(numRaysPerTile)
    {
        for ( u32 i=0; i<2; i++ )
        {
            m_rayBoxQueue[i]  = nullptr;
            m_rayBoxBuffer[i] = nullptr;
            m_leafQueue[i]  = nullptr;
            m_leafBuffer[i] = nullptr;
        }

    #if RL_PRINT_STATS
        printf("\n--- Tracer allocations ---\n\n");
    #endif
        
        for ( u32 i=0; i<2; i++ )
        {
            m_rayBoxQueue[i]  = new DeviceBuffer(sizeof(Store<RayBox>));
            m_rayBoxBuffer[i] = new DeviceBuffer(numRayBoxQueries*sizeof(RayBox));
            m_leafQueue[i]  = new DeviceBuffer(sizeof(Store<RayLeaf>));
            m_leafBuffer[i] = new DeviceBuffer(numLeafQueries*sizeof(RayLeaf));
        #if RL_PRINT_STATS
            printf("RayBoxQueries %d, count %d, size %.3fmb\n", i, numRayBoxQueries, (float)m_rayBoxBuffer[i]->size()/1024/1024);
            printf("RayLeafQueries %d, count %d, size %.3fmb\n", i, numLeafQueries, (float)m_leafBuffer[i]->size()/1024/1024);
        #endif
        }

        // Assign device buffers to queue elements ptr
        for ( u32 i=0; i<2; i++ )
        {
            COPY_PTR_TO_DEVICE_ASYNC(m_rayBoxQueue[i], m_rayBoxBuffer[i], Store<RayBox>, m_elements);
            COPY_VALUE_TO_DEVICE_ASYNC(m_rayBoxQueue[i], numRayBoxQueries, Store<RayBox>, m_max, sizeof(u32));
            COPY_PTR_TO_DEVICE_ASYNC(m_leafQueue[i], m_leafBuffer[i], Store<RayLeaf>, m_elements);
            COPY_VALUE_TO_DEVICE_ASYNC(m_leafQueue[i], numLeafQueries, Store<RayLeaf>, m_max, sizeof(u32));
        }

        u64 totalMemory = 0;
        for ( u32 i=0; i<2; i++ )
        {
            totalMemory += m_rayBoxBuffer[i]->size() + m_rayBoxQueue[i]->size();
            totalMemory += m_leafBuffer[i]->size() + m_leafQueue[i]->size();
        }
        
        printf("Total: %.3fmb\n", (float)totalMemory/1024/1024);
        printf("\n--- End Tracer allocations ---\n\n");
    }

    Tracer::~Tracer()
    {
        for ( u32 i=0; i<2; ++i )
        {
            delete m_leafQueue[i];
            delete m_leafBuffer[i];
            delete m_rayBoxQueue[i];
            delete m_rayBoxQueue[i];
        }
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
        Store<RayLeaf>* leafQueues[]    = { m_leafQueue[0]->ptr<Store<RayLeaf>>(), m_leafQueue[1]->ptr<Store<RayLeaf>>() };
        vec3* rayOris                   = trq->m_oris->ptr<vec3>(); // Not const, because first origin is derived from eye
        const vec3* rayDirs             = trq->m_dirs->ptr<const vec3>();
        const BvhNode* bvhNodes         = scn->m_bvhTree->ptr<const BvhNode>();
        const Face* faces               = scn->m_faces->ptr<const Face>();
        const FaceCluster* faceClusters = scn->m_faceClusters->ptr<const FaceCluster>();
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
                          (Store<RayBox>**) rbQueues, (Store<RayLeaf>**) leafQueues,
                           rayOris, rayDirs,
                           bvhNodes, faces, faceClusters,
                           meshData,
                           hitResults[0]);

            m_profiler.stop("Tile " + to_string(kTile++));
            totalRays -= numRaysThisTile;
        }

        m_profiler.endProfile("Trace");

        return ERROR_ALL_FINE;
    }

}