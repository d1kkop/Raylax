#include "ReylaxCommon.h"
#include "Reylax.h"
#include "Reylax_internal.h"

#define NUM_RAYBOX_QUEUES 32
#define NUM_LEAF_QUEUES 32

#define BLOCK_THREADS 256


namespace Reylax
{
    GLOBAL void ResetRayBoxKernel(Store<RayBox>* rayBoxQueue)
    {
        rayBoxQueue->m_top = 0;
    }

    GLOBAL void ResetRayLeafKernel(Store<RayBox>* leafQueue)
    {
        leafQueue->m_top = 0;
    }

    GLOBAL void ResetRayFaceKernel(Store<RayFace>* rayFaceQueue)
    {
        rayFaceQueue->m_top = 0;
        ResetRayLeafKernel<<<1, 1>>>( nullptr );
    }

    GLOBAL void SetInitialRbQueriesKernel(u32 numRays, Store<RayBox>* rayBoxQueue)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= numRays ) return;
        RayBox* rb = rayBoxQueue->getNew(1);
        rb->node = 0;
        rb->ray  = i;
    }

    GLOBAL void RayBoxKernel(vec3 eye,
                             mat3 orient,
                             u32 numRayBoxes,
                             Store<RayBox>* rayBoxQueueIn,
                             Store<RayBox>* rayBoxQueueOut,
                             Store<RayBox>* leafQueue,
                             const vec3* rayDirs,
                             const BvhNode* bvhNodes)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= numRayBoxes ) return;

        RayBox* rb    = rayBoxQueueIn->get(i);
        vec3 d        = orient * (rayDirs + rb->ray)[0];
        const BvhNode* node = bvhNodes + rb->node;

        vec3 cp  = node->cp;
        vec3 hs  = node->hs;
        vec3 invDir(1.f/d.x, 1.f/d.y, 1.f/d.z);

        if ( BoxRayIntersect(cp-hs, cp+hs, eye, invDir) )
        {
            if ( node->isLeaf() )
            {
                RayBox* rbLeaf = leafQueue->getNew(1);
                *rbLeaf = *rb;
            }
            else
            {
                // both must be valid
                assert(RL_VALID_INDEX(node->left) && RL_VALID_INDEX(node->right));
                // create 2 new ray/box queries
                RayBox* rbNew = rayBoxQueueOut->getNew(2);
                rbNew->ray  = rb->ray;
                rbNew->node = node->left;
                (rbNew+1)->ray  = rb->ray;
                (rbNew+1)->node = node->right;
            }
        }
    }

    GLOBAL void RayBoxQueueStichKernel(u32 numInnerQueues,
                                       Store<RayBox>* rayBoxQueueIn,
                                       Store<RayBox>* rayBoxQueueOut,
                                       Store<RayBox>* leafQueue,
                                       Ray* rayBuffer,
                                       const BvhNode* bvhNodes)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= numInnerQueues ) return;
    }

    GLOBAL void LeafExpandKernel(u32 numLeafs,
                                 Store<RayBox>* leafQueue,
                                 Store<RayFace>* rayFaceQueue,
                                 const BvhNode* bvhNodes,
                                 const FaceCluster* faceClusters)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= numLeafs ) return;
        RayBox* rb = leafQueue->get(i);
        const BvhNode* node = bvhNodes + rb->node;
        assert(node->isLeaf());
        u32 numFaces = node->numFaces();
        RayFace* rf  = rayFaceQueue->getNew(numFaces);
        u32 ray      = rb->ray;
        for ( u32 k=0; k<numFaces; ++k )
        {
            rf[k].ray  = ray;
            rf[k].face = node->getFace(faceClusters, k);
        }
    }

    GLOBAL void FaceTestKernel(u32 numRayFaceQueries,
                               Ray* rayBuffer,
                               Store<RayFace>* rayFaceQueue,
                               FaceCluster* faceClusters,
                               Face* faceBuffer,
                               RayFaceHitCluster* hitResultClusters,
                               MeshData* meshData)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= numRayFaceQueries ) return;
        RayFace* rf = rayFaceQueue->get(i);
        Ray* ray    = rayBuffer + rf->ray;
        Face* face  = faceBuffer + rf->face;
        vec3 d = ray->d;
        vec3 o = ray->o;
        float u, v;
        float dist = FaceRayIntersect(face, o, d, meshData, u, v);
        if ( dist != FLT_MAX )
        {
            RayFaceHitCluster* hitCluster = hitResultClusters + rf->ray;
            u32 curHitIdx = atomicAdd2<u32>(&hitCluster->count, 1);
            assert( curHitIdx < TRACER_MAX_HITS_PER_RAY );
            if ( curHitIdx < TRACER_MAX_HITS_PER_RAY )
            {
                RayFaceHitResult* result = hitCluster->results + curHitIdx;
                result->u = u;
                result->v = v;
                result->dist = dist;
                result->face = face;
                for ( u32 i=0; i<3; ++i )
                {
                    result->ro[i] = o[i];
                    result->rd[i] = d[i];
                }
            }
        }
    }

    GLOBAL void FindClosestHit(u32 numRays,
                               RayFaceHitCluster* hitResultClusters,
                               RayFaceHitResult* finalResults)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= numRays ) return;
        RayFaceHitCluster* hitCluster = hitResultClusters + i;
        RayFaceHitResult* results     = hitCluster->results;
        u32 count   = hitCluster->count;
        float fDist = FLT_MAX;
        u32 closest = RL_INVALID_INDEX;
        for ( u32 k=0; k<count; ++k )
        {
            RayFaceHitResult* result = results + k;
            if ( result->dist < fDist )
            {
                fDist = result->dist;
                closest = k;
            }
        }
        RayFaceHitResult* finResult = finalResults + i;
        if ( RL_VALID_INDEX(closest) )
        {
            memcpy( finResult, results + closest, sizeof(RayFaceHitResult) );
        }
        else
        {
            finResult->face = nullptr;
        }
    }
}



extern "C"
{
    using namespace Reylax;


    u32 rlResetRayBoxQueue(Store<RayBox>* queue)
    {
        if (!queue) return ERROR_INVALID_PARAMETER;
    #if RL_CUDA
        ResetRayBoxKernel<<<1, 1>>>(queue);
    #else
        ResetRayBoxKernel(queue);
    #endif
        return ERROR_ALL_FINE;
    }

    u32 rlResetRayLeafQueue(Store<RayBox>* queue)
    {
        if (!queue) return ERROR_INVALID_PARAMETER;
    #if RL_CUDA
        ResetRayLeafKernel<<<1, 1>>>(queue);
    #else
        ResetRayLeafKernel(queue);
    #endif
        return ERROR_ALL_FINE;
    }

    u32 rlResetRayFaceQueue(Store<RayFace>* queue)
    {
        if ( !queue ) return ERROR_INVALID_PARAMETER;
    #if RL_CUDA
        ResetRayFaceKernel<<<1, 1>>>(queue);
    #else
        ResetRayFaceKernel(queue);
    #endif
        return ERROR_ALL_FINE;
    }

    u32 rlSetInitialRbQueries(u32 numRays, Store<RayBox>* queue)
    {
        if ( numRays==0 || !queue ) return ERROR_INVALID_PARAMETER;
        dim3 blocks  ((numRays + BLOCK_THREADS-1)/BLOCK_THREADS);
        dim3 threads (BLOCK_THREADS);
    #if RL_CUDA
        SetInitialRbQueriesKernel<<<blocks, threads>>>( numRays, queue );
    #else
        emulateCpu(BLOCK_THREADS, blocks, threads, [=]()
        {
            SetInitialRbQueriesKernel(numRays, queue);
        });
    #endif
        return ERROR_ALL_FINE;
    }

    u32 rlRayBox(const float* eye3,
                 const float* orient3x3,
                 u32 numRayBoxes,
                 Store<RayBox>* rayBoxQueueIn,
                 Store<RayBox>* rayBoxQueueOut,
                 Store<RayBox>* leafQueue,
                 const vec3* rayDirs,
                 const BvhNode* bvhNodes)
    {
        if ( numRayBoxes==0 || !rayBoxQueueIn || !rayBoxQueueOut || !leafQueue || !rayDirs || !bvhNodes )
        {
            return ERROR_INVALID_PARAMETER;
        }
        vec3 eye = *(vec3*)eye3;
        mat3 orient = *(mat3*)orient3x3;
        dim3 blocks  ((numRayBoxes + BLOCK_THREADS-1)/BLOCK_THREADS);
        dim3 threads (BLOCK_THREADS);
    #if RL_CUDA
        RayBoxKernel<<< blocks, threads >>>( eye, orient, numRayBoxes, rayBoxQueueIn, rayBoxQueueOut, leafQueue, rayDirs, bvhNodes );
    #else
        emulateCpu(BLOCK_THREADS, blocks, threads, [=]()
        { 
            RayBoxKernel( eye, orient, numRayBoxes, rayBoxQueueIn, rayBoxQueueOut, leafQueue, rayDirs, bvhNodes );
        });
    #endif

        return ERROR_ALL_FINE;
    }

    u32 rlExpandLeafs(u32 numRayLeafQueries,
                      Store<RayBox>* leafQueue,
                      Store<RayFace>* rayFaceQueue,
                      const BvhNode* bvhNodes,
                      const FaceCluster* faceClusters)
    {
        if ( numRayLeafQueries==0 || !leafQueue || !rayFaceQueue || !bvhNodes || !faceClusters )
        {
            return ERROR_INVALID_PARAMETER;
        }
        dim3 blocks  ((numRayLeafQueries + BLOCK_THREADS-1)/BLOCK_THREADS);
        dim3 threads (BLOCK_THREADS);
    #if RL_CUDA
        LeafExpandKernel<<< blocks, threads >>>(numRayLeafQueries, leafQueue, rayFaceQueue, bvhNodes, faceClusters );
    #else
        emulateCpu(BLOCK_THREADS, blocks, threads, [=]()
        {
            LeafExpandKernel( numRayLeafQueries, leafQueue, rayFaceQueue, bvhNodes, faceClusters );
        });
    #endif

        return ERROR_ALL_FINE;
    }
}