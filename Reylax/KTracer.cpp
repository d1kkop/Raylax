#include "ReylaxCommon.h"

#define BLOCK_THREADS 256
#define NUM_RAYBOX_QUEUES 32
#define NUM_LEAF_QUEUES 32
#define MARCH_EPSILON 0.0001f

#define DBG_QUERIES 0

#define DBG_PIXEL (256*128+140)

#if DBG_QUERIES
    #define DBG_QUERIES_BEGIN( str ) printf("\n--- %s ---\n", (str))
    #define DBG_QUERIES_IN( itr, num ) printf("Itr: %d, in %d\n", (itr), (num))
    #define DBG_QUERIES_OUT( numOut, numOther ) printf("Out: %d, newQueue %d\n", (numOut),(numOther) )
    #define DBG_QUERIES_END( str) printf("\n--- End %s queries---\n", (str) )
#else
    #define DBG_QUERIES_BEGIN( str )
    #define DBG_QUERIES_IN( itr, num )
    #define DBG_QUERIES_OUT( numOut, numOther )
    #define DBG_QUERIES_END( str)
#endif


namespace Reylax
{
    DEVICE CONSTANT TracerContext ct;

    void UpdateTraceContext(const TracerContext& newCt, bool wait)
    {
        SetSymbol( ct, &newCt );
    }

    FDEVICE void gridSync()
    {
    #if RL_CUDA
        cudaDeviceSynchronize();
    #endif
    }

    DEVICE void QueueRay(const float* ori3, const float* dir3)
    {
        // Only if we hit root box of tree, an intersection can take place.
        vec3 o = *(vec3*)ori3;
        vec3 d = *(vec3*)dir3;
        vec3 invd   = vec3(1.f/d.x, 1.f/d.y, 1.f/d.z);
        float kDist = BoxRayIntersect(ct.bMin, ct.bMax, o, invd);
        if ( kDist != FLT_MAX )
        {
            Ray* ray = ct.rayPayload->getNew(1);
            ray->o = o;
            ray->d = d;
            ray->invd = invd;
            ray->sign[0] = d.x > 0 ? 1 : 0;
            ray->sign[1] = d.y > 0 ? 1 : 0;
            ray->sign[2] = d.z > 0 ? 1 : 0;
            PointBox* pb = ct.pbQueues[0]->getNew(1);
            pb->point    = o + d*(kDist+MARCH_EPSILON);
            pb->localId  = bIdx.x * bDim.x + tIdx.x;
            pb->node     = 0;
            pb->ray      = (u32)(ray - ct.rayPayload->m_elements);
        }
    }

    GLOBAL void PointBoxKernel()
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= ct.pbQueues[ct.queueIn]->m_top ) return;

        PointBox* pb = ct.pbQueues[ct.queueIn]->get(i);
        const BvhNode* node = ct.bvhNodes + pb->node;

        if ( node->isLeaf() )
        {
            if ( node->numFaces() != 0 )
            {
                RayLeaf* rl = ct.leafQueues[0]->getNew(1);
                rl->pb      = *pb;
                rl->faceIdx = 0;
            }
            else
            {
                const vec3* bounds  = &node->bMin;
                float distToBox = FLT_MAX;
                vec3 pnt = pb->point;
                Ray* ray = ct.rayPayload->get(pb->ray);
                u32 nextBoxId   = SelectNextBox(bounds, ct.sides + node->right*6, ray->sign, pnt, ray->invd, distToBox);
                if ( nextBoxId != RL_INVALID_INDEX )
                {
                    vec3 dir = ray->d;
                    PointBox* pbNew = ct.pbQueues[ct.queueOut]->getNew(1);
                    pbNew->point    = pnt + dir*(distToBox+MARCH_EPSILON); // TODO check epsilon
                    pbNew->node     = nextBoxId;
                    pbNew->localId  = pb->localId;
                    pbNew->ray      = pb->ray;
                }
            }
        }
        else if ( PointInAABB(pb->point, node->bMin, node->bMax) )
        {
            // both must be valid
            assert(RL_VALID_INDEX(node->left) && RL_VALID_INDEX(BVH_GET_INDEX(node->right)));
            PointBox* pbNew = ct.pbQueues[ct.queueOut]->getNew(1);
            *pbNew           = *pb;
            u32 spAxis      = BVH_GET_AXIS(node->right);
            float s         = (node->bMax[spAxis] + node->bMin[spAxis])*.5f;
            if ( pb->point[spAxis] < s )
            {
                pbNew->node = node->left;
            }
            else
            {
                pbNew->node = BVH_GET_INDEX( node->right );
            }
        }
    }

    GLOBAL void RayLeafKernel()
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= ct.leafQueues[ct.queueIn]->m_top ) return;

        RayLeaf* leaf    = ct.leafQueues[ct.queueIn]->get(i);
        const auto* node = ct.bvhNodes + leaf->pb.node;
        const Ray*  ray  = ct.rayPayload->get(leaf->pb.ray);

        assert( node->isLeaf() );

        const vec3& pnt  = leaf->pb.point;
        const vec3& dir  = ray->d;

        u32 numFaces = node->numFaces();
        u32 faceIdx  = leaf->faceIdx;
        u32 loopCnt  = numFaces;// TODO _min(numFaces - faceIdx, 4U);  // Change this to reduce number of writes back to global memory at the cost of more idling threads.

        HitResult* result = ct.hitResults + leaf->pb.localId;
        float fDist = result->dist;
        float fU    = -1.f;
        float fV    = -1.f;
        const Face* closestFace = nullptr;

        const Face* faces            = ct.faces;
        const FaceCluster* fclusters = ct.faceClusters;

        // Check N faces, increase number for less writes back to memory at the cost of more idling threads (divergent execution).
    #pragma unroll
        for ( u32 k=0; k<loopCnt; ++k )
        {
            const Face* face = faces + node->getFace(fclusters, faceIdx);
            float u, v;
            float dist = FaceRayIntersect(face, pnt, dir, ct.meshDataPtrs, u, v);
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
        if ( fU != -1.f )
        {
            result->u = fU;
            result->v = fV;
            result->dist = fDist;
            result->face = closestFace;
            result->ray  = leaf->pb.ray;
            vec3 ori = ray->o;
        #pragma unroll
            for ( u32 i=0; i<3; ++i )
            {
                result->ro[i] = ori[i];
                result->rd[i] = dir[i];
            }
        }

        // If there are still faces left to process, queue new Ray/Leaf item
        if ( faceIdx < numFaces )
        {
            PointBox pb = leaf->pb;
            leaf = ct.leafQueues[ct.queueOut]->getNew(1);
            leaf->pb = pb;
            leaf->faceIdx = faceIdx;
        }
        else // If no faces left to process, march ray to next box
        {
            const vec3* bounds  = &node->bMin;
            float distToBox = FLT_MAX;
            u32 nextBoxId   = SelectNextBox( bounds, ct.sides + node->right*6, ray->sign, pnt, ray->invd, distToBox );
            if ( nextBoxId != RL_INVALID_INDEX && result->dist > distToBox )
            {
                PointBox* pb    = ct.pbQueues[0]->getNew(1);
                pb->point       = pnt + dir*(distToBox+MARCH_EPSILON); // TODO check epsilon
                pb->node        = nextBoxId;
                pb->localId     = leaf->pb.localId;
                pb->ray         = leaf->pb.ray;
            }
        }
    }

    GLOBAL void SetupRays(u32 numRays, u32 tileOffset)
    {
        u32 localId = bIdx.x * bDim.x + tIdx.x;
        if ( localId >= numRays ) return;
        u32 globalId = tileOffset + localId;
        ct.setupCb( globalId, localId );
    }

    GLOBAL void ResetHitResults(u32 numRays)
    {
        u32 localId = bIdx.x * bDim.x + tIdx.x;
        if ( localId >= numRays ) return;
        ct.hitResults[localId].dist = FLT_MAX;
    }

    GLOBAL void HitCallbackKernel(u32 numRays, u32 tileOffset)
    {
        u32 localId = bIdx.x * bDim.x + tIdx.x;
        if ( localId >= numRays ) return;
       // if ( ct.hitResults[localId].dist == FLT_MAX ) return;
        u32 globalId = tileOffset + localId;
        const HitResult& hit = ct.hitResults[localId];
        //Ray* ray = ct.rayPayload->get( hit.ray );
        ct.hitCb( globalId, localId, ct.curDepth, hit, ct.meshDataPtrs );
    }

    template <class QIn, class Qout, class Func>
    DEVICE void DoQueries(const char* queries, u32 numRays, Store<QIn>** queryQueues, Store<Qout>* remainderQueue, Func f)
    {
        ct.queueIn  = 0;
        ct.queueOut = 1;
        remainderQueue->m_top = 0;

        DBG_QUERIES_BEGIN(queries);

        // if no queries left to solve, quit
        while ( queryQueues[ct.queueIn]->m_top!=0 )
        {
            // ensure top out is set to zero
            queryQueues[ct.queueOut]->m_top = 0;

            DBG_QUERIES_IN(i, queryQueues[ct.queueIn]->m_top);

            // execute all rb queries from queue-in and generate new to queue-out or leaf-queue
            dim3 blocks  ((queryQueues[ct.queueIn]->m_top + BLOCK_THREADS-1)/BLOCK_THREADS);
            dim3 threads (BLOCK_THREADS);
            RL_KERNEL_CALL(BLOCK_THREADS, blocks, threads, f);
            gridSync();

            DBG_QUERIES_OUT(queryQueues[ct.queueOut]->m_top, remainderQueue->m_top);

            ct.queueIn  = (ct.queueIn+1)&1;
            ct.queueOut = (ct.queueOut+1)&1;
        }

        DBG_QUERIES_END(queries);
    }

    GLOBAL void TileKernel(u32 numRays, u32 tileOffset)
    {
        ct.curDepth = 0;
        ct.rayPayload->m_top  = 0;
        ct.pbQueues[0]->m_top = 0;

        dim3 blocks  ((numRays + BLOCK_THREADS-1)/BLOCK_THREADS);
        dim3 threads (BLOCK_THREADS);
        RL_KERNEL_CALL( BLOCK_THREADS, blocks, threads, SetupRays, numRays, tileOffset );
        gridSync();

        while ( ct.pbQueues[ct.queueIn]->m_top != 0 && ct.curDepth < ct.maxDepth )
        {
            RL_KERNEL_CALL( BLOCK_THREADS, blocks, threads, ResetHitResults, numRays );
            gridSync();

            // Solve all intersections iteratively
            u32 numIters = 0;
            while ( ct.pbQueues[ct.queueIn]->m_top != 0 )
            {
                // Iterate Point-box queries until queues are empty
                DoQueries("Point-box", numRays, ct.pbQueues, ct.leafQueues[0], PointBoxKernel);

                // Iterate Ray/leaf queries until queues are empty
                DoQueries("Ray-leaf", numRays, ct.leafQueues, ct.pbQueues[0], RayLeafKernel);

                ct.queueIn = 0;
                numIters++;
            } // End while there are still point-box queries
            printf("-- Num iters for a single tile %d --\n", numIters );

            // TODO: For now, for each ray in tile, execute hit result (whether it was hit or not)
            dim3 blocks  ((numRays + BLOCK_THREADS-1)/BLOCK_THREADS);
            dim3 threads (BLOCK_THREADS);
            ct.queueIn  = 0;
            ct.queueOut = 1;
            ct.pbQueues[0]->m_top = 0;
            RL_KERNEL_CALL(BLOCK_THREADS, blocks, threads, HitCallbackKernel, numRays, tileOffset);
            gridSync();
            
            ++ct.curDepth;
        }
    }

}