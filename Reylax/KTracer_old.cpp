#include "ReylaxCommon.h"
#include "Reylax.h"
#include "Reylax_internal.h"

#define BLOCK_THREADS 256
#define NUM_RAYBOX_QUEUES 32
#define NUM_LEAF_QUEUES 32

#define DBG_RB_QUERIES 0
#define DBG_RL_QUERIES 0
#define DBG_RF_QUERIES 0


namespace Reylax
{
    DEVICE vec3 g_eye;
    DEVICE mat3 g_orient;

    GLOBAL void PerRayInitializationKernel(u32 numRays, Store<RayBox>* rayBoxQueue)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= numRays ) return;
        RayBox* rb = rayBoxQueue->getNew(1);
        rb->node = 0;
        rb->ray  = i;
    }

    GLOBAL void RayBoxKernel(u32 numRayBoxes,
                             Store<RayBox>* rayBoxQueueIn,
                             Store<RayBox>* rayBoxQueueOut,
                             Store<RayLeaf>* leafQueue,
                             const vec3* rayDirs,
                             const BvhNode* bvhNodes)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= numRayBoxes ) return;

        RayBox* rb    = rayBoxQueueIn->get(i);
        vec3 d        = g_orient * (rayDirs + rb->ray)[0];
        const BvhNode* node = bvhNodes + rb->node;

        vec3 o   = g_eye;
        vec3 cp  = node->cp;
        vec3 hs  = node->hs;
        vec3 invDir(1.f/d.x, 1.f/d.y, 1.f/d.z);

        if ( BoxRayIntersect(cp-hs, cp+hs, o, invDir) )
        {
            if ( node->isLeaf() )
            {
                if ( node->numFaces() != 0 )
                {
                    RayLeaf* rl = leafQueue->getNew(1);
                    rl->faceIdx = 0;
                    rl->node = rb->node;
                    rl->ray  = rb->ray;
                }
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

    GLOBAL void RayLeafKernel(u32 numLeafs,
                              Store<RayLeaf>* leafQueueIn,
                              Store<RayLeaf>* leafQueueOut,
                              const vec3* rayOris,
                              const vec3* rayDirs,
                              const BvhNode* bvhNodes,
                              const Face* faces,
                              const FaceCluster* faceClusters,
                              const MeshData* const* meshData,
                              HitResult* hitResults)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= numLeafs ) return;
        RayLeaf* rl = leafQueueIn->get(i);
        const BvhNode* leaf = bvhNodes + rl->node;
        assert(leaf->isLeaf() && rl->faceIdx < leaf->numFaces());
        u32 ray = rl->ray;
        u32 faceIdx  = rl->faceIdx;
        const Face* face = faces + leaf->getFace(faceClusters, faceIdx);
        const vec3& o = rayOris[ray];
        const vec3& d = rayDirs[ray];
        float u, v;
        float dist = FaceRayIntersect(face, o, d, meshData, u, v);
        HitResult* result = hitResults + ray;
        if ( dist < result->dist )
        {
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
        if ( ++faceIdx < leaf->numFaces() )
        {
             rl = leafQueueOut->getNew(1);
             rl->faceIdx = faceIdx;
             rl->node = rl->node;
             rl->ray  = rl->ray;
        }
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
                               const Store<RayFace>* rayFaceQueue,
                               const FaceCluster* faceClusters,
                               const vec3* rayOris,
                               const vec3* rayDirs,
                               const Face* faces,
                               HitCluster* hitResultClusters,
                               const MeshData* const* meshData)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= numRayFaceQueries ) return;
        RayFace* rf = rayFaceQueue->get(i);
        const Face* face  = faces + rf->face;
        vec3 d = rayDirs[rf->ray];
        vec3 o = rayOris[rf->ray];
        float u, v;
        float dist = FaceRayIntersect(face, o, d, meshData, u, v);
        if ( dist != FLT_MAX )
        {
            HitCluster* hitCluster = hitResultClusters + rf->ray;
            u32 curHitIdx = atomicAdd2<u32>(&hitCluster->count, 1);
            assert( curHitIdx < TRACER_MAX_HITS_PER_RAY );
            if ( curHitIdx < TRACER_MAX_HITS_PER_RAY )
            {
                HitResult* result = hitCluster->results + curHitIdx;
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
                               HitCluster* hitResultClusters,
                               HitResult* finalResults)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= numRays ) return;
        HitCluster* hitCluster = hitResultClusters + i;
        HitResult* results     = hitCluster->results;
        u32 count   = hitCluster->count;
        float fDist = FLT_MAX;
        u32 closest = RL_INVALID_INDEX;
        for ( u32 k=0; k<count; ++k )
        {
            HitResult* result = results + k;
            if ( result->dist < fDist )
            {
                fDist = result->dist;
                closest = k;
            }
        }
        HitResult* finResult = finalResults + i;
        if ( RL_VALID_INDEX(closest) )
        {
            memcpy( finResult, results + closest, sizeof(HitResult) );
        }
        else
        {
            finResult->face = nullptr;
        }
    }

    GLOBAL void TileKernel(u32 numRays,
                           vec3 eye,
                           mat3 orient,
                           Store<RayBox>** rbQueues,
                           Store<RayLeaf>** leafQueues,
                           const vec3* rayOris,
                           const vec3* rayDirs,
                           const BvhNode* bvhNodes,
                           const Face* faces,
                           const FaceCluster* faceClusters,
                           const MeshData* const* meshData,
                           HitResult** hitResults, u32 numResults)
    {
        g_eye    = eye;
        g_orient = orient;

        rbQueues[0]->m_top = 0;
        leafQueues[0]->m_top   = 0;

        // Set up initial Ray/box queries
        dim3 blocks  ((numRays + BLOCK_THREADS-1)/BLOCK_THREADS);
        dim3 threads (BLOCK_THREADS);
        RL_KERNEL_CALL(BLOCK_THREADS, blocks, threads, PerRayInitializationKernel, numRays, rbQueues[0]);

        // Iterate Ray/box queries until queues are empty
        {
        #if DBG_RB_QUERIES
            printf("\n--- Ray/box queries ---\n\n");
        #endif

            for ( u32 i=0; /*no stop cond*/; ++i )
            {
                u32 inQueue  = i%2;
                u32 outQueue = (i+1)%2;

                // stop if all rb queries are done
                if ( rbQueues[inQueue]->m_top==0 )
                {
                    break;
                }

                // ensure top out is set to zero
                rbQueues[outQueue]->m_top = 0;

            #if DBG_RB_QUERIES
                printf("--- RB iteration %d ---\n", i);
                printf("Num RB In/Out Before %d/%d\n", rbQueues[inQueue]->m_top, rbQueues[outQueue]->m_top);
                printf("Num LeafQ %d Before\n", leafQueue->m_top);
            #endif

                // execute all rb queries from queue-in and generate new to queue-out or leaf-queue
                u32 numRayBoxes = rbQueues[inQueue]->m_top;
                blocks = dim3((numRayBoxes + BLOCK_THREADS-1)/BLOCK_THREADS);
                RL_KERNEL_CALL(BLOCK_THREADS, blocks, threads, RayBoxKernel, numRayBoxes, rbQueues[inQueue], rbQueues[outQueue], leafQueues[0], rayDirs, bvhNodes);
                cudaDeviceSynchronize();

            #if DBG_RB_QUERIES
                printf("Num RB In/Out After %d/%d\n", rbQueues[inQueue]->m_top, rbQueues[outQueue]->m_top);
                printf("Num LeafQ %d After\n", leafQueue->m_top);
            #endif
            }

        #if DBG_RB_QUERIES
            printf("\n--- End Ray/box queries ---\n");
        #endif
        }

        //// Leaf queries
        //{
        //    u32 numLeafQueries = leafQueue->m_top;
        //#if DBG_RL_QUERIES
        //    printf("\n--- Begin Leaf queries %d ---", numLeafQueries);
        //#endif

        //    blocks = dim3((numLeafQueries + BLOCK_THREADS-1)/BLOCK_THREADS);
        //    RL_KERNEL_CALL(BLOCK_THREADS, blocks, threads, LeafExpandKernel, numLeafQueries, leafQueue, rayFaceQueue, bvhNodes, faceClusters);
        //    cudaDeviceSynchronize();

        //#if DBG_RL_QUERIES
        //    printf("--- End Leaf queries %d ---\n", numLeafQueries);
        //#endif
        //}

        //// Ray/face queries
        //{
        //    u32 numRayFace = rayFaceQueue->m_top;
        //#if DBG_RL_QUERIES
        //    printf("\n--- Begin Ray/face queries %d ---", numRayFace);
        //#endif

        //    blocks = dim3((numRayFace + BLOCK_THREADS-1)/BLOCK_THREADS);
        //    RL_KERNEL_CALL(BLOCK_THREADS, blocks, threads, FaceTestKernel, numRayFace, rayFaceQueue, faceClusters, rayOris, rayDirs, faces, hitResultClusters, meshData);
        //    cudaDeviceSynchronize();

        //#if DBG_RL_QUERIES
        //    printf("--- End Ray/face queries %d ---\n", numRayFace);
        //#endif
        //}
    }
    
}