#include "ReylaxCommon.h"

#define BLOCK_THREADS 256
#define NUM_RAYBOX_QUEUES 32
#define NUM_LEAF_QUEUES 32

#define DBG_RB_QUERIES 0
#define DBG_RL_QUERIES 0

#define MARCH_EPSILON 0.001f


namespace Reylax
{
    DEVICE vec3 g_eye;
    DEVICE mat3 g_orient;
    DEVICE vec3 g_bMin;
    DEVICE vec3 g_bMax;
    DEVICE HitCallback g_hitCallback;

    GLOBAL void PerRayInitializationKernel(u32 numRays,
                                           char* raySigns, vec3* rayOris, const vec3* rayDirs, 
                                           Store<PointBox>* pointBoxQueue, HitResult* hitResults)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= numRays ) return;
        hitResults[i].dist = FLT_MAX;
        rayOris[i] = g_eye;
        const vec3& d = rayDirs[i];
        vec3 invD(1.f/d.x, 1.f/d.y, 1.f/d.z);
        float dist = BoxRayIntersect( g_bMin, g_bMax, g_eye, invD );
        if ( dist != FLT_MAX )
        {
            PointBox* pb = pointBoxQueue->getNew(1);
            pb->node  = 0;
            pb->ray   = i;
            pb->point = g_eye + (dist+MARCH_EPSILON)*d;
            char* signs = raySigns + i*3;
            signs[0] = d.x > 0 ? 0 : 1;
            signs[1] = d.y > 0 ? 0 : 1;
            signs[2] = d.z > 0 ? 0 : 1;
        }
    }

    GLOBAL void PointBoxKernel(u32 numPoints,
                               Store<PointBox>* pointBoxQueueIn,
                               Store<PointBox>* pointBoxQueueOut,
                               Store<RayLeaf>* leafQueue,
                               const BvhNode* bvhNodes)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= numPoints ) return;

        PointBox* pb = pointBoxQueueIn->get(i);
        const BvhNode* node = bvhNodes + pb->node;

        if ( node->isLeaf() )
        {
            if ( node->numFaces() != 0 )
            {
                RayLeaf* rl = leafQueue->getNew(1);
                rl->faceIdx = 0;
                rl->node = pb->node;
                rl->ray  = pb->ray;
            }
        }
        else
        {
            // both must be valid
            assert(RL_VALID_INDEX(BVH_GET_INDEX(node->left)) && RL_VALID_INDEX(BVH_GET_INDEX(node->right)));
            PointBox* pbNew = pointBoxQueueOut->getNew(1);
            *pbNew = *pb;

            const BvhNode* nodeT = bvhNodes + node->left;
            if ( PointInAABB(pb->point, nodeT->bMin, nodeT->bMax) )
            {
                pbNew->node = BVH_GET_INDEX( node->left );
            }
            else
            {
                pbNew->node = BVH_GET_INDEX( node->right );
            }
        }
    }


    GLOBAL void RayLeafKernel(u32 numLeafs,
                              Store<RayLeaf>* leafQueueIn,
                              Store<RayLeaf>* leafQueueOut,
                              Store<PointBox>* pointBoxQueue,
                              const vec3* rayOris,
                              const vec3* rayDirs,
                              const char*  raySigns,
                              const BvhNode* bvhNodes,
                              const Face* faces,
                              const FaceCluster* faceClusters,
                              const u32* sides,
                              const MeshData* const* meshData,
                              HitResult* hitResults)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= numLeafs ) return;
        RayLeaf* rl = leafQueueIn->get(i);
        u32 node    = rl->node;
        const BvhNode* leaf = bvhNodes + node;
        u32 numFaces = leaf->numFaces();
        assert(leaf->isLeaf() && rl->faceIdx < numFaces);
        u32 ray = rl->ray;
        const vec3& o = rayOris[ray];
        const vec3& d = rayDirs[ray];
        u32 faceIdx   = rl->faceIdx;
        u32 loopCnt   = _min(numFaces - rl->faceIdx, 4U);  // Change this to reduce number of writes back to global memory at the cost of more idling threads
        HitResult* result = hitResults + ray;
        float fDist = result->dist;                     // Continue comparing with last store hit result distance
        float fU = -1.f;
        float fV = -1.f;
        const Face* closestFace = nullptr;

        // Check N faces, increase number for less writes back to memory at the cost of more idling threads (divergent execution).
    #pragma unroll
        for ( u32 k=0; k<loopCnt; ++k )
        {
            const Face* face = faces + leaf->getFace(faceClusters, faceIdx);
            float u, v;
            float dist = FaceRayIntersect(face, o, d, meshData, u, v);
            if ( dist < fDist )
            {
                fU = u;
                fV = v;
                fDist = dist;
                closestFace = face;
            }
            ++faceIdx;
        }

        // New closer distance was found, write back to global memory
        if ( fU != -1.f ) // TODO check if dist is less than dist to box edge
        {
            result->u = fU;
            result->v = fV;
            result->dist = fDist;
            result->face = closestFace;
        #pragma unroll
            for ( u32 i=0; i<3; ++i )
            {
                result->ro[i] = o[i];
                result->rd[i] = d[i];
            }
        }

        // If there are still faces left to process, queue new Ray/Leaf item
        if ( faceIdx < numFaces )
        {
            rl = leafQueueOut->getNew(1);
            rl->faceIdx = faceIdx;
            rl->node = node;
            rl->ray  = ray;
        }
        else // If no faces left to process, march ray to next box
        {
            const vec3* bounds = &leaf->bMin;
            u32 sideIdx = leaf->right; // No need to do GET_INDEX as no spAxis is stored in leaf_right
            vec3 invDir(1.f/d.x, 1.f/d.y, 1.f/d.z);
            float distToBox = 0.f;
            u32 nextBoxId = SelectNextBox( bounds, sides + sideIdx*6 , raySigns + ray*3, o, invDir, distToBox );
            if ( result->dist > distToBox )
            {
                PointBox* pb = pointBoxQueue->getNew(1);
                pb->ray   = ray;
                pb->point = o + d*(distToBox+MARCH_EPSILON); // TODO check epsilon
                pb->node  = nextBoxId;
            }
        }
    }

    GLOBAL void HitCallbackKernel(u32 numRays,
                                  u32 tileOffset,
                                  const HitResult* hitResults,
                                  const MeshData* const* meshData,
                                  float* rayOris, float* rayDirs,
                                  HitCallback cb)
    {
        u32 localId = bIdx.x * bDim.x + tIdx.x;
        if ( localId >= numRays ) return;
        u32 globalId = tileOffset + localId;
        cb( globalId, localId, hitResults[localId], meshData, rayOris, rayDirs );
    }


    GLOBAL void TileKernel(u32 numRays,
                           vec3 eye,
                           mat3 orient,
                           Store<PointBox>** pbQueues,
                           Store<RayLeaf>** leafQueues,
                           char* raySigns,
                           vec3* rayOris,
                           const vec3* rayDirs,
                           const BvhNode* bvhNodes,
                           const Face* faces,
                           const FaceCluster* faceClusters,
                           const u32* sides,
                           const MeshData* const* meshData,
                           HitResult* hitResults)
    {
        g_eye    = eye;
        g_orient = orient;
        g_bMin = bvhNodes->bMin;
        g_bMax = bvhNodes->bMax;

        pbQueues[0]->m_top = 0;
        leafQueues[0]->m_top = 0;

        // Do per ray (in tile) initialization
        dim3 blocks  ((numRays + BLOCK_THREADS-1)/BLOCK_THREADS);
        dim3 threads (BLOCK_THREADS);
        RL_KERNEL_CALL(BLOCK_THREADS, blocks, threads, PerRayInitializationKernel, numRays, 
                       raySigns, rayOris, rayDirs, 
                       pbQueues[0], hitResults);

        u32 numIters = 0;
        while ( pbQueues[0]->m_top != 0 )
        {
            // Iterate Ray/box queries until queues are empty
            {
            #if DBG_RB_QUERIES
                printf("\n--- Point-box queries ---\n\n");
            #endif

                for ( u32 i=0; /*no stop cond*/; ++i )
                {
                    u32 inQueue  = i%2;
                    u32 outQueue = (i+1)%2;

                    // stop if all rb queries are done
                    if ( pbQueues[inQueue]->m_top==0 )
                    {
                        break;
                    }

                    // ensure top out is set to zero
                    pbQueues[outQueue]->m_top = 0;

                #if DBG_RB_QUERIES
                    printf("\n--- RB iteration %d ---\n", i);
                    printf("Num points in %d\n", pbQueues[inQueue]->m_top);
                #endif

                    // execute all rb queries from queue-in and generate new to queue-out or leaf-queue
                    u32 numPointBoxes = pbQueues[inQueue]->m_top;
                    blocks = dim3((numPointBoxes + BLOCK_THREADS-1)/BLOCK_THREADS);
                    RL_KERNEL_CALL(BLOCK_THREADS, blocks, threads, PointBoxKernel, numPointBoxes, pbQueues[inQueue], pbQueues[outQueue], leafQueues[0], bvhNodes);
                    cudaDeviceSynchronize();

                #if DBG_RB_QUERIES
                    printf("Num points out  %d\n", pbQueues[outQueue]->m_top);
                    printf("Num leafs out %d\n", leafQueues[0]->m_top);
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

                // set pb queue to 0 again as ray may not hit a single face inside a box, in such case the ray needs to check the next box.
                pbQueues[0]->m_top = 0;

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
                    printf("Num leaf in %d\n", leafQueues[inQueue]->m_top);
                #endif

                    // execute all ray/leaf queries from queue-in and generate new to queue-out or leaf-queue
                    u32 numRayLeafs = leafQueues[inQueue]->m_top;
                    blocks = dim3((numRayLeafs + BLOCK_THREADS-1)/BLOCK_THREADS);
                    RL_KERNEL_CALL(BLOCK_THREADS, blocks, threads, RayLeafKernel,
                                   numRayLeafs, leafQueues[inQueue], leafQueues[outQueue], pbQueues[0],
                                   rayOris, rayDirs, raySigns, bvhNodes, faces, faceClusters, sides,
                                   meshData, hitResults);
                    cudaDeviceSynchronize();

                #if DBG_RL_QUERIES
                    printf("Num leaf out %d\n", leafQueues[outQueue]->m_top);
                #endif
                }

            #if DBG_RL_QUERIES
                printf("\n--- End Ray-leaf queries ---\n");
            #endif
            }

            ++numIters;
        } // End while there are still point-box queries

        printf("-- Num iters for a single tile %d\n", numIters );
    }
    
}