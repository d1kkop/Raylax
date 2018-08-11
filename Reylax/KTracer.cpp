#include "ReylaxCommon.h"

#define BLOCK_THREADS 256
#define NUM_RAYBOX_QUEUES 32
#define NUM_LEAF_QUEUES 32

#define DBG_RB_QUERIES 1
#define DBG_RL_QUERIES 1


namespace Reylax
{
    DEVICE vec3 g_eye;
    DEVICE mat3 g_orient;

    GLOBAL void PerRayInitializationKernel(u32 numRays, Store<RayBox>* rayBoxQueue, HitResult* hitResults)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= numRays ) return;
        RayBox* rb = rayBoxQueue->getNew(1);
        rb->node = 0;
        rb->ray  = i;
        hitResults[i].dist = FLT_MAX;
    }

    GLOBAL void RayBoxKernel(u32 numRayBoxes,
                             Store<RayBox>* rayBoxQueueIn,
                             Store<RayBox>* rayBoxQueueOut,
                             Store<RayLeaf>* leafQueue,
                             vec3* rayOris,
                             const vec3* rayDirs,
                             const BvhNode* bvhNodes)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= numRayBoxes ) return;

        RayBox* rb    = rayBoxQueueIn->get(i);
        vec3 d        = g_orient * (rayDirs + rb->ray)[0];
        const BvhNode* node = bvhNodes + rb->node;

        // set rayOri from eye
        (rayOris + rb->ray)[0] = g_eye;

        vec3 cp  = node->cp;
        vec3 hs  = node->hs;
        vec3 invDir(1.f/d.x, 1.f/d.y, 1.f/d.z);

        if ( BoxRayIntersect(cp-hs, cp+hs, g_eye, invDir) )
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
        u32 node = rl->node;
        const BvhNode* leaf = bvhNodes + node;
        u32 numFaces = leaf->numFaces();
        assert(leaf->isLeaf() && rl->faceIdx < numFaces);
        u32 ray = rl->ray;
        u32 faceIdx = rl->faceIdx;
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
        #pragma unroll
            for ( u32 i=0; i<3; ++i )
            {
                result->ro[i] = o[i];
                result->rd[i] = d[i];
            }
        }
        if ( ++faceIdx < numFaces )
        {
             rl = leafQueueOut->getNew(1);
             rl->faceIdx = faceIdx;
             rl->node = node;
             rl->ray  = ray;
        }
    }

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
                           HitResult* hitResults)
    {
        g_eye    = eye;
        g_orient = orient;

        rbQueues[0]->m_top = 0;
        leafQueues[0]->m_top = 0;

        // Do per ray (in tile) initialization
        dim3 blocks  ((numRays + BLOCK_THREADS-1)/BLOCK_THREADS);
        dim3 threads (BLOCK_THREADS);
        RL_KERNEL_CALL(BLOCK_THREADS, blocks, threads, PerRayInitializationKernel, numRays, rbQueues[0], hitResults);

        // Iterate Ray/box queries until queues are empty
        {
        #if DBG_RB_QUERIES
            printf("\n--- Ray-box queries ---\n\n");
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
                printf("\n--- RB iteration %d ---\n", i);
                printf("Num RayIn %d\n", rbQueues[inQueue]->m_top);
            #endif

                // execute all rb queries from queue-in and generate new to queue-out or leaf-queue
                u32 numRayBoxes = rbQueues[inQueue]->m_top;
                blocks = dim3((numRayBoxes + BLOCK_THREADS-1)/BLOCK_THREADS);
                RL_KERNEL_CALL(BLOCK_THREADS, blocks, threads, RayBoxKernel, numRayBoxes, rbQueues[inQueue], rbQueues[outQueue], leafQueues[0], rayOris, rayDirs, bvhNodes);
                cudaDeviceSynchronize();

            #if DBG_RB_QUERIES
                printf("Num RayOut  %d\n", rbQueues[outQueue]->m_top);
                printf("Num RayLeaf %d\n", leafQueues[0]->m_top);
            #endif
            }

        #if DBG_RB_QUERIES
            printf("\n--- End Ray-box queries ---\n");
        #endif
        }

        // Iterate Ray/leaf queries until queues are empty
        {
        #if DBG_RL_QUERIES
            printf("\n--- Ray-leaf queries ---\n\n");
        #endif

            for ( u32 i=0; /*no stop cond*/; ++i )
            {
                u32 inQueue  = i%2;
                u32 outQueue = (i+1)%2;

                // stop if all ray/leaf queries are done
                if ( leafQueues[inQueue]->m_top==0 )
                {
                    break;
                }

                // ensure top out is set to zero
                leafQueues[outQueue]->m_top = 0;

            #if DBG_RL_QUERIES
                printf("\n--- Leaf iteration %d ---\n", i);
                printf("Num Leaf-In %d\n", leafQueues[inQueue]->m_top);
            #endif

                // execute all ray/leaf queries from queue-in and generate new to queue-out or leaf-queue
                u32 numRayLeafs = leafQueues[inQueue]->m_top;
                blocks = dim3((numRayLeafs + BLOCK_THREADS-1)/BLOCK_THREADS);
                RL_KERNEL_CALL(BLOCK_THREADS, blocks, threads, RayLeafKernel,
                               numRayLeafs, leafQueues[inQueue], leafQueues[outQueue],
                               rayOris, rayDirs, bvhNodes, faces, faceClusters, meshData, hitResults);
                cudaDeviceSynchronize();

            #if DBG_RL_QUERIES
                printf("Num LeafOut %d\n", leafQueues[outQueue]->m_top);
            #endif
            }

        #if DBG_RL_QUERIES
            printf("\n--- End Ray-leaf queries ---\n");
        #endif
        }
    }
    
}