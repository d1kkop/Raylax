#include "ReylaxCommon.h"

#define MARCH_EPSILON 0.00001f
#define NUM_FACE_ITERS 4U

#define POINTBOX_STOP_QUOTEM 0U
#define RAYLEAF_STOP_QUOTEM 0U

#define DBG_QUERIES 0
#define DBG_SHOW_TILE_ITERS 1


#if DBG_QUERIES
    #define DBG_QUERIES_BEGIN( str ) u32 i=0; printf("\n--- %s Queries Begin ---\n", (str))
    #define DBG_QUERIES_IN( itr, num ) printf("Itr: %d, in %d", (itr), (num))
    #define DBG_QUERIES_OUT( out ) printf(" out: %d\n", (out))
    #define DBG_QUERIES_END( str, total ) printf("\n--- %s End, num iters: %d ---\n", (str), (total) )
    #define DBG_QUERIES_RESTART( num ) printf("New Pb queries: %d \n", (num) )
#else
    #define DBG_QUERIES_BEGIN( str )
    #define DBG_QUERIES_IN( itr, num )
    #define DBG_QUERIES_OUT( out )
    #define DBG_QUERIES_END( str, total )
    #define DBG_QUERIES_RESTART( num )
#endif


namespace Reylax
{
    DEVICE CONSTANT TracerContext ct;
    

#if !RL_CUDA_DYN
    TracerContext h_ct;
#else
    #define h_ct ct
#endif

    void UpdateTraceContext(const TracerContext& newCt, bool wait)
    {
        SetSymbol( ct, &newCt );
    #if !RL_CUDA_DYN
        h_ct = newCt;
    #endif
    }

    DEVICE_DYN void dynamicSync()
    {
    #if RL_CUDA_DYN
        cudaDeviceSynchronize();
    #endif
    }

    DEVICE void QueueRay(const float* ori3, const float* dir3)
    {
        return;
        // Only if we hit root box of tree, an intersection can take place.
        vec3 o = *(vec3*)ori3;
        vec3 d = *(vec3*)dir3;
        vec3 invd   = vec3(1.f/d.x, 1.f/d.y, 1.f/d.z);
        float kDist = BoxRayIntersect(ct.bMin, ct.bMax, o, invd);
        if ( kDist != FLT_MAX )
        {
            u32 i = bIdx.x * bDim.x + tIdx.x;
            Ray* ray = ct.rayPayload->getNew(i, 1);
            ray->o = o;
            ray->d = d;
            ray->invd = invd;
            ray->sign[0] = d.x > 0 ? 1 : 0;
            ray->sign[1] = d.y > 0 ? 1 : 0;
            ray->sign[2] = d.z > 0 ? 1 : 0;
            PointBox* pb = ct.pbQueues[ct.pbQueueIn]->getNew(i, 1);
            pb->point    = o + d*(kDist+MARCH_EPSILON);
            pb->localId  = i;
            pb->node     = 0;
            pb->ray      = (u32)(ray - ct.rayPayload->m_elements);
        }
    }

    DEVICE CONSTANT QueueRayFptr queueRayFptr = QueueRay;
    QueueRayFptr GetQueueRayFptr()
    {
        QueueRayFptr rval;
        GetSymbol( &rval, queueRayFptr );
        return rval;
    }

    GLOBAL void PointBoxKernel(u32 queueLength)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= queueLength ) return;

        PointBox* pb = ct.pbQueues[ct.pbQueueIn]->get(ct.id2Queue, i);
        const BvhNode* node = ct.bvhNodes + pb->node;

        if ( node->isLeaf() )
        {
            if ( node->numFaces() != 0 )
            {
                RayLeaf* rl = ct.leafQueues[ct.rlQueueIn]->getNew(i, 1);
                rl->pb      = *pb;
                rl->faceIdx = 0;
            }
            else
            {
                const vec3* bounds  = &node->bMin;
                float distToBox = FLT_MAX;
                vec3 pnt = pb->point;
                Ray* ray = ct.rayPayload->getFromBase(pb->ray);
                u32 nextBoxId   = SelectNextBox(bounds, ct.sides + node->right*6, ray->sign, pnt, ray->invd, distToBox);
                if ( nextBoxId != RL_INVALID_INDEX )
                {
                    vec3 dir = ray->d;
                    PointBox* pbNew = ct.pbQueues[ct.pbQueueOut]->getNew(i, 1);
                    pbNew->point    = pnt + dir*(distToBox+MARCH_EPSILON);
                    pbNew->node     = nextBoxId;
                    pbNew->localId  = pb->localId;
                    pbNew->ray      = pb->ray;
                }
            }
        }
        else /* if ( PointInAABB(pb->point, node->bMin, node->bMax) ) */ // Should no longer be necessary as the marching point always progresses, so eventually an 'invalid' box will be selected and the trace quits.
        {
            // both must be valid
            assert(RL_VALID_INDEX(node->left) && RL_VALID_INDEX(BVH_GET_INDEX(node->right)));
            PointBox* pbNew = ct.pbQueues[ct.pbQueueOut]->getNew(i, 1);
            *pbNew          = *pb;
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

    GLOBAL void RayLeafKernel(u32 queueLength)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= queueLength ) return;

        RayLeaf* leaf    = ct.leafQueues[ct.rlQueueIn]->get(ct.id2Queue, i);
        const auto* node = ct.bvhNodes + leaf->pb.node;
        const Ray*  ray  = ct.rayPayload->getFromBase(leaf->pb.ray);

        assert( node->isLeaf() );

        const vec3& pnt  = leaf->pb.point;
        const vec3& dir  = ray->d;

        u32 numFaces = node->numFaces();
        u32 faceIdx  = leaf->faceIdx;
        u32 loopCnt  = _min(numFaces - faceIdx, NUM_FACE_ITERS);

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
            leaf = ct.leafQueues[ct.rlQueueOut]->getNew(i, 1);
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
                PointBox* pb    = ct.pbQueues[ct.pbQueueIn]->getNew(i, 1);
                pb->point       = pnt + dir*(distToBox+MARCH_EPSILON);
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
        assert( ct.setupCb );
        printf("Setup Fptr = %p\n", ct.setupCb );
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
        if ( ct.hitResults[localId].dist == FLT_MAX ) return;
        u32 globalId = tileOffset + localId;
        const HitResult& hit = ct.hitResults[localId];
        //Ray* ray = ct.rayPayload->get( hit.ray );
        assert ( ct.hitCb );
        printf("Fptr = %p\n", ct.hitCb );
        ct.hitCb( globalId, localId, ct.curDepth, hit, ct.meshDataPtrs );
    }

    template <class QIn>
    GLOBAL void UpdateToInnerQueueKernel(u32 queueLength, QIn* qIn, byte* id2Queue)
    {
        u32 localId = bIdx.x * bDim.x + tIdx.x;
        if ( localId >= queueLength ) return;
    #if _DEBUG
        bool inserted=false;
    #endif
        u32 nextTop=0;
        for ( u32 i=0; i<RL_NUMMER_INNER_QUEUES; i++ )
        {
            nextTop += qIn->getTop(i);
            if ( localId < nextTop )
            {
                assert(i<256);
                id2Queue[localId] = i;
            #if _DEBUG
                inserted=true;
            #endif
                break;
            }
        }
        assert(inserted);
    }

    template <class QIn>
    DEVICE_DYN u32 UpdateToSingleQueue(QIn& qIn, byte* id2Queue)
    {
    #if RL_CUDA_DYN
        u32 qlength = qIn->updateToSingleQueue();
    #if RL_USE_INNER_QUEUES
        dim3 blocks  ((qlength + RL_BLOCK_THREADS-1)/RL_BLOCK_THREADS);
        dim3 threads (RL_BLOCK_THREADS);
        RL_KERNEL_CALL(RL_BLOCK_THREADS, blocks, threads, UpdateToInnerQueueKernel, qlength, qIn, id2Queue);
        dynamicSync();
    #endif
        return qlength;
    #else
        u32 queueLength = 0;
        Reylax::hostOrDeviceCpy(&queueLength, ((char*)qIn) + offsetof(Store<PointBox>, m_top), 4, cudaMemcpyDeviceToHost, false);
        return queueLength;
    #endif
    }

    template <class QIn>
    DEVICE_DYN void SetQueueTopZero(QIn& dQueue)
    {
    #if RL_CUDA_DYN
        dQueue->resetTop();
    #else
        u32 zero = 0;
        Reylax::hostOrDeviceCpy(((char*)dQueue) + offsetof(Store<PointBox>, m_top), &zero, 4, cudaMemcpyHostToDevice, true);
    #endif
    }

    template <class QIn, class Qout, class Func>
    DEVICE_DYN void DoQueries(const char* queries, u32& qIn, u32& qOut, u32 queueLength, u32 quotumThreshold, Store<QIn>** queryQueues, Store<Qout>* remainderQueue, Func f)
    {
        assert( qIn != qOut );

        DBG_QUERIES_BEGIN(queries);

        // if no queries left to solve, quit
        while ( queueLength > quotumThreshold )
        {
            // ensure top out is set to zero
            SetQueueTopZero( queryQueues[qOut] );

            DBG_QUERIES_IN(i++, queueLength);

            // execute all rb queries from queue-in and generate new to queue-out or leaf-queue
            dim3 blocks  ((queueLength + RL_BLOCK_THREADS-1)/RL_BLOCK_THREADS);
            dim3 threads (RL_BLOCK_THREADS);
            RL_KERNEL_CALL(RL_BLOCK_THREADS, blocks, threads, f, queueLength);
            dynamicSync();

            queueLength = UpdateToSingleQueue( queryQueues[qOut], h_ct.id2Queue );
            DBG_QUERIES_OUT(queueLength);

            qIn  = (qIn+1)&1;
            qOut = (qOut+1)&1;
            assert( qIn != qOut );
        }

        DBG_QUERIES_END(queries, i);
    }

    // TODO helper kernel , can be put in TileKernel when can be run on GPU
    FDEVICE_DYN void PrepareKernel()
    {
        h_ct.curDepth   = 0;
        h_ct.pbQueueIn  = 0;
        h_ct.pbQueueOut = 1;
        h_ct.rlQueueIn  = 0;
        h_ct.rlQueueOut = 1;
        SetQueueTopZero( h_ct.rayPayload );
        SetQueueTopZero( h_ct.pbQueues[0] );
        SetQueueTopZero( h_ct.pbQueues[1] );
        SetQueueTopZero( h_ct.leafQueues[0] );
        SetQueueTopZero( h_ct.leafQueues[1] );
    }

    GLOBAL_DYN void TileKernel(u32 numRays, u32 tileOffset)
    {
        PrepareKernel();

        dim3 threads (RL_BLOCK_THREADS);
        dim3 blocks  ((numRays + RL_BLOCK_THREADS-1)/RL_BLOCK_THREADS);
        RL_KERNEL_CALL( RL_BLOCK_THREADS, blocks, threads, SetupRays, numRays, tileOffset );
        dynamicSync();

        u32 queueLength = UpdateToSingleQueue( h_ct.pbQueues[h_ct.pbQueueIn], h_ct.id2Queue );
        while ( queueLength > POINTBOX_STOP_QUOTEM && h_ct.curDepth < h_ct.maxDepth )
        {
            blocks.x = ((numRays + RL_BLOCK_THREADS-1)/RL_BLOCK_THREADS);
            RL_KERNEL_CALL( RL_BLOCK_THREADS, blocks, threads, ResetHitResults, numRays );
            dynamicSync();

            // Solve all intersections iteratively
            u32 numIters = 0;
            while ( queueLength > POINTBOX_STOP_QUOTEM )
            {
                // Iterate Point-box queries until queues are empty
                DoQueries("Point-box", h_ct.pbQueueIn, h_ct.pbQueueOut, queueLength, POINTBOX_STOP_QUOTEM, h_ct.pbQueues, h_ct.leafQueues[h_ct.rlQueueIn], PointBoxKernel);

                // Iterate Ray/leaf queries until queues are empty
                queueLength = UpdateToSingleQueue( h_ct.leafQueues[h_ct.rlQueueIn], h_ct.id2Queue );
                DoQueries("Ray-leaf", h_ct.rlQueueIn, h_ct.rlQueueOut, queueLength, RAYLEAF_STOP_QUOTEM, h_ct.leafQueues, h_ct.pbQueues[h_ct.pbQueueIn], RayLeafKernel);

                queueLength = UpdateToSingleQueue( h_ct.pbQueues[h_ct.pbQueueIn], h_ct.id2Queue );
                DBG_QUERIES_RESTART( queueLength );
                numIters++;
            } // End while there are still point-box queries

        #if DBG_SHOW_TILE_ITERS
            printf("-- Num iters tile %d --\n", numIters );
        #endif

            // TODO: For now, for each ray in tile, execute hit result (whether it was hit or not)
            blocks.x = ((numRays + RL_BLOCK_THREADS-1)/RL_BLOCK_THREADS);
            RL_KERNEL_CALL(RL_BLOCK_THREADS, blocks, threads, HitCallbackKernel, numRays, tileOffset);
            dynamicSync();
            
            ++h_ct.curDepth;
            queueLength = 0; // TODO hard quit no recursion
        }
    }

}