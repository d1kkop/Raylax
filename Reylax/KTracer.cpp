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

    DEVICE void FirstRays(u32 globalId, u32 localId, Store<PointBox>* pbIn);
    DEVICE void TraceCallback(u32 globalId, u32 localId, u32 depth, const HitResult& hit, const MeshData* const* meshPtrs);
    

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

    DEVICE void QueueRay(const float* ori3, const float* dir3, Store<PointBox>* pbIn)
    {
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
            PointBox* pb = pbIn->getNew(i, 1);
            pb->point    = o + d*(kDist+MARCH_EPSILON);
            pb->localId  = i;
            pb->node     = 0;
            pb->ray      = (u32)(ray - ct.rayPayload->m_elements);
        }
    }

    GLOBAL void PointBoxKernel(u32 queueLength, Store<PointBox>* pbIn, Store<PointBox>* pbOut, Store<RayLeaf>* leafOut)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= queueLength ) return;

        PointBox* pb = pbIn->get(ct.id2Queue, i);
        const BvhNode* node = ct.bvhNodes + pb->node;

        if ( node->isLeaf() )
        {
            if ( node->numFaces() != 0 )
            {
                RayLeaf* rl = leafOut->getNew(i, 1);
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
                    PointBox* pbNew = pbOut->getNew(i, 1);
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
            PointBox* pbNew = pbOut->getNew(i, 1);
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

    GLOBAL void RayLeafKernel(u32 queueLength, Store<RayLeaf>* leafIn, Store<RayLeaf>* leafOut, Store<PointBox>* pbOut)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= queueLength ) return;

        RayLeaf* leaf    = leafIn->get(ct.id2Queue, i);
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
            leaf = leafOut->getNew(i, 1);
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
                PointBox* pb    = pbOut->getNew(i, 1);
                pb->point       = pnt + dir*(distToBox+MARCH_EPSILON);
                pb->node        = nextBoxId;
                pb->localId     = leaf->pb.localId;
                pb->ray         = leaf->pb.ray;
            }
        }
    }

    GLOBAL void SetupRays(u32 numRays, u32 tileOffset, Store<PointBox>* pbIn)
    {
        u32 localId = bIdx.x * bDim.x + tIdx.x;
        if ( localId >= numRays ) return;
        u32 globalId = tileOffset + localId;
        FirstRays( globalId, localId, pbIn );
        /*assert( ct.setupCb );
        printf("Setup Fptr = %p\n", ct.setupCb );
        ct.setupCb( globalId, localId );*/
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
        TraceCallback( globalId, localId, ct.curDepth, hit, ct.meshDataPtrs );
        //Ray* ray = ct.rayPayload->get( hit.ray );
        /*assert ( ct.hitCb );
        printf("Fptr = %p\n", ct.hitCb );
        ct.hitCb( globalId, localId, ct.curDepth, hit, ct.meshDataPtrs );*/
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
        return GetQueueTop(qIn);
    #endif
    }

    template <class QIn>
    DEVICE_DYN u32 GetQueueTop(QIn& dQueue)
    {
    #if RL_CUDA_DYN
        return dQueue->updateToSingleQueue();
    #else
        u32 queueLength = 0;
        u32 err = hostOrDeviceCpy(&queueLength, ((char*)dQueue) + offsetof(Store<PointBox>, m_top), 4, cudaMemcpyDeviceToHost, false);
        assert(err==0);
        return queueLength;
    #endif
    }

    template <class QIn>
    DEVICE_DYN void SetQueueTop(QIn& dQueue, u32 val=0)
    {
    #if RL_CUDA_DYN
        dQueue->resetTop();
    #else
        u32 err = hostOrDeviceCpy(((char*)dQueue) + offsetof(Store<PointBox>, m_top), &val, 4, cudaMemcpyHostToDevice, true);
        assert(err==0);
    #endif
    }

    template <class QIn, class Qout, class Func>
    DEVICE_DYN void DoQueries(const char* queriesName, u32 queueLength, u32 quotumThreshold, 
                              Store<QIn>*& queueIn, Store<QIn>*& queueOut, Store<Qout>* remainderQueue, Func f)
    {
        assert( queueIn != queueOut );

        DBG_QUERIES_BEGIN(queriesName);

        // if no queries left to solve, quit
        while ( queueLength > quotumThreshold )
        {
            // ensure top out is set to zero
            SetQueueTop( queueOut );

            DBG_QUERIES_IN(i++, queueLength);

            // execute all rb queries from queue-in and generate new to queue-out or leaf-queue
            dim3 blocks  ((queueLength + RL_BLOCK_THREADS-1)/RL_BLOCK_THREADS);
            dim3 threads (RL_BLOCK_THREADS);
            RL_KERNEL_CALL(RL_BLOCK_THREADS, blocks, threads, f, queueLength, queueIn, queueOut, remainderQueue);
            dynamicSync();

            queueLength = UpdateToSingleQueue( queueOut, h_ct.id2Queue );
            DBG_QUERIES_OUT(queueLength);

            // Swap queues
            auto* qTemp = queueIn;
            queueIn  = queueOut;
            queueOut = qTemp;
            assert( queueIn != queueOut );
        }

        DBG_QUERIES_END(queriesName, i);
    }

    // TODO helper kernel , can be put in TileKernel when can be run on GPU
    FDEVICE_DYN void PrepareKernel()
    {
        h_ct.curDepth = 0;
        SetQueueTop( h_ct.rayPayload );
        SetQueueTop( h_ct.pbQueueIn );
        SetQueueTop( h_ct.pbQueueOut );
        SetQueueTop( h_ct.leafQueueIn );
        SetQueueTop( h_ct.leafQueueOut );
    }

    GLOBAL_DYN void TileKernel(u32 numRays, u32 tileOffset)
    {
        PrepareKernel();

        dim3 threads (RL_BLOCK_THREADS);
        dim3 blocks  ((numRays + RL_BLOCK_THREADS-1)/RL_BLOCK_THREADS);
        RL_KERNEL_CALL( RL_BLOCK_THREADS, blocks, threads, SetupRays, numRays, tileOffset, h_ct.pbQueueIn );
        dynamicSync();

        u32 queueLength = UpdateToSingleQueue( h_ct.pbQueueIn, h_ct.id2Queue );
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
                DoQueries("Point-box", queueLength, POINTBOX_STOP_QUOTEM, h_ct.pbQueueIn, h_ct.pbQueueOut, h_ct.leafQueueIn, PointBoxKernel);

                // Iterate Ray/leaf queries until queues are empty
                queueLength = UpdateToSingleQueue( h_ct.leafQueueIn, h_ct.id2Queue );
                DoQueries("Ray-leaf", queueLength, RAYLEAF_STOP_QUOTEM, h_ct.leafQueueIn, h_ct.leafQueueOut, h_ct.pbQueueIn, RayLeafKernel );

                queueLength = UpdateToSingleQueue( h_ct.pbQueueIn, h_ct.id2Queue );
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

// ------------------------------------ user trace -------------------------------------------------------------



namespace Reylax
{
    __align__(8)
    struct TraceData
    {
        vec3 eye;
        mat3 orient;
        vec3* rayDirs;
        u32*  pixels;
    };

    DEVICE TraceData TD;


    void UpdateTraceData(const vec3& eye, mat3& orient, vec3* rays, u32* pixels)
    {
        TraceData td;
        td.eye = eye;
        td.orient  = orient;
        td.rayDirs = rays;
        td.pixels  = pixels;
        Reylax::SetSymbol(TD, &td);
    }

    template <class T>
    DEVICE T Interpolate(const HitResult& hit, const MeshData* const* meshPtrs, u32 dataIdx)
    {
        assert(meshPtrs);
        assert(dataIdx < VERTEX_DATA_COUNT);
        const MeshData* mesh = meshPtrs[hit.face->w];
        const T* vd  = (const T*)(mesh->vertexData[dataIdx]);
        const T& vd1 = vd[hit.face->x];
        const T& vd2 = vd[hit.face->y];
        const T& vd3 = vd[hit.face->z];
        float u = hit.u;
        float v = hit.v;
        float w = 1-(u+v);
        return w*vd1 + u*vd2 + v*vd3;
    }

    DEVICE u32 rgba(const vec4& c)
    {
        u32 r = (u32)(c.x*255.f);
        u32 g = (u32)(c.y*255.f);
        u32 b = (u32)(c.z*255.f);
        u32 a = (u32)(c.w*255.f);
        if ( r > 255 ) r = 255;
        if ( g > 255 ) g = 255;
        if ( b > 255 ) b = 255;
        if ( a > 255 ) a = 255;
        return (a<<24)|(r<<16)|(g<<8)|(b);
    }

    DEVICE u32 single(float f)
    {
        u32 r = (u32)(f*255.f);
        if ( r > 255 ) r = 255;
        return r;
    }

    DEVICE void FirstRays(u32 globalId, u32 localId, Store<PointBox>* pbIn)
    {
        //printf("yeeey %d %d\n", globalId, localId);
        vec3 dir = /*TD.orient **/ TD.rayDirs[globalId];
        vec3 ori = /*TD.eye;*/ vec3(0, 0, -2.5f);
        QueueRay(&ori.x, &dir.x, pbIn);
        //  QueueRayFunc( &ori.x, &dir.x ); 
    }

    DEVICE void TraceCallback(u32 globalId, u32 localId, u32 depth,
                              const HitResult& hit,
                              const MeshData* const* meshPtrs)
    {
        vec3 n = Interpolate<vec3>(hit, meshPtrs, VERTEX_DATA_NORMAL);
        n = normalize(n);
        TD.pixels[globalId] = single(abs(n.z)) << 16;
    }
}