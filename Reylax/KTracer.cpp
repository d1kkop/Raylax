#include "ReylaxCommon.h"

#define MARCH_EPSILON 0.00001f
#define NUM_FACE_ITERS 32U

#define POINTBOX_STOP_QUOTEM 1000U
#define RAYLEAF_STOP_QUOTEM 0U

#define DBG_QUERIES 0
#define DBG_SHOW_TILE_ITERS 0


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

    FDEVICE void QueueRay(vec3 o, vec3 d, Store<PointBox>* pbIn)
    {
        // Only if we hit root box of tree, an intersection can take place.
        vec3 invd   = vec3(1.f/d.x, 1.f/d.y, 1.f/d.z);
        float kDist = BoxRayIntersect(ct.bMin, ct.bMax, o, invd);
        if ( kDist != FLT_MAX )
        {
            u32 i = bIdx.x * bDim.x + tIdx.x;
            Ray r2;
            r2.o = o;
            r2.d = d;
            r2.invd = invd;
            r2.sign[0] = d.x > 0 ? 1 : 0;
            r2.sign[1] = d.y > 0 ? 1 : 0;
            r2.sign[2] = d.z > 0 ? 1 : 0;
            Ray* ray = ct.rayPayload->getNew(i, 1);
            *ray = r2;

            PointBox pb2;
            pb2.point    = o + d*(kDist+MARCH_EPSILON);
            pb2.localId  = i;
            pb2.node     = 0;
            pb2.ray      = (u32)(ray - ct.rayPayload->m_elements);
            PointBox* pb = pbIn->getNew(i, 1);
            *pb = pb2;
        }
    }

    GLOBAL void PointBoxKernel(u32 queueLength, Store<PointBox>* pbIn, Store<PointBox>* pbOut, Store<RayLeaf>* leafOut)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= queueLength ) return;

        //PointBox pb  = *pbIn->get(ct.id2Queue, i);
        SHARED PointBox pbs[RL_BLOCK_THREADS];
        pbs[tIdx.x] = *pbIn->get(ct.id2Queue, i);
      //  PointBox* pb = &pbs[tIdx.x];
        //SHARED Ray rds[RL_BLOCK_THREADS];
        //rds[tIdx.x] = *ct.rayPayload->getFromBase(pbs[tIdx.x].ray);
        //Ray* ray = &rds[tIdx.x];
        Ray* ray = ct.rayPayload->getFromBase(pbs[tIdx.x].ray); 
        vec3 d   = ray->d;
        vec3 invd = ray->invd;

        while ( true )
        {
            const BvhNode* node = ct.bvhNodes + pbs[tIdx.x].node;
            if ( node->isLeaf() )
            {
                if ( node->numFaces() != 0 )
                {
                    RayLeaf* rl = leafOut->getNew(i, 1);
                    rl->pb      = pbs[tIdx.x];
                    rl->faceIdx = 0;
                    return;
                }
                else
                {
                    float distToBox;
                    u32 nextBoxId = SelectNextBox( &node->bMin, ct.sides + node->right*6, ray->sign, pbs[tIdx.x].point, invd, distToBox );
                    if ( nextBoxId != RL_INVALID_INDEX )
                    {
                        pbs[tIdx.x].point += d*(distToBox+MARCH_EPSILON);
                        pbs[tIdx.x].node = nextBoxId;
                    }
                    else
                    {
                        return;
                    }
                }
            }
            else
            {
                // both must be valid
                assert(RL_VALID_INDEX(node->left) && RL_VALID_INDEX(BVH_GET_INDEX(node->right)));
                u32 spAxis  = BVH_GET_AXIS(node->right);
            #if !BVH_HAS_SPLIT
                float s = (node->bMax[spAxis] + node->bMin[spAxis])*.5f;
            #else
                float s = node->split;
            #endif
                if ( pbs[tIdx.x].point[spAxis] < s ) { pbs[tIdx.x].node = node->left; }
                else { pbs[tIdx.x].node = BVH_GET_INDEX(node->right); }
            }
        }
    }

    GLOBAL void RayLeafKernel(u32 queueLength, Store<RayLeaf>* leafIn, Store<RayLeaf>* leafOut, Store<PointBox>* pbOut)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= queueLength ) return;

        SHARED RayLeaf leaf[RL_BLOCK_THREADS];
        leaf[tIdx.x] = *leafIn->get(ct.id2Queue, i);

        const auto* node = ct.bvhNodes + leaf[tIdx.x].pb.node;
        const Ray*  ray  = ct.rayPayload->getFromBase(leaf[tIdx.x].pb.ray);

        assert( node->isLeaf() );

        const vec3& pnt  = leaf[tIdx.x].pb.point;
        const vec3& dir  = ray->d;

        u32 numFaces = node->numFaces();
        u32 faceIdx  = leaf[tIdx.x].faceIdx;
        u32 loopCnt  = _min(numFaces - faceIdx, NUM_FACE_ITERS);

        HitResult* result = ct.hitResults + leaf[tIdx.x].pb.localId;
        float fDist = result->dist;
        float fU    = -1.f;
        float fV    = -1.f;
        const Face* closestFace = nullptr;

        const Face* faces            = ct.faces;
        const FaceCluster* fclusters = ct.faceClusters;

        // Check N faces, increase number for less writes back to memory at the cost of more idling threads (divergent execution).
  //  #pragma unroll
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
            result->ray  = leaf[tIdx.x].pb.ray;
            vec3 ori = ray->o;
        #pragma unroll
            for ( u32 i=0; i<3; ++i )
            {
                result->ro[i] = ori[i];
                result->rd[i] = dir[i];
            }
        }

       //  If there are still faces left to process, queue new Ray/Leaf item
        if ( faceIdx < numFaces )
        {
            RayLeaf* leaf = leafOut->getNew(i, 1);
            leaf->pb = leaf[tIdx.x].pb;
            leaf->faceIdx = faceIdx;
        }
        else // If no faces left to process, march ray to next box
        {
            float distToBox;
            u32 nextBoxId   = SelectNextBox( &node->bMin, ct.sides + node->right*6, ray->sign, pnt, ray->invd, distToBox );
            if ( nextBoxId != RL_INVALID_INDEX && result->dist > distToBox )
            {
                PointBox* pb    = pbOut->getNew(i, 1);
                pb->point       = pnt + dir*(distToBox+MARCH_EPSILON);
                pb->node        = nextBoxId;
                pb->localId     = leaf[tIdx.x].pb.localId;
                pb->ray         = leaf[tIdx.x].pb.ray;
            }
        }
    }

    GLOBAL void SolveRay(u32 queueLength, Store<PointBox>* pbIn)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= queueLength ) return;

        PointBox pb = *pbIn->get(ct.id2Queue, i);
        Ray ray     = *ct.rayPayload->getFromBase( pb.ray );
        HitResult result = *(ct.hitResults + pb.localId);

        while ( true )
        {
            BvhNode node = *(ct.bvhNodes + pb.node);
            if ( node.isLeaf() )
            {
                u32 numFaces = node.numFaces();
                if ( numFaces != 0 )
                {
                    u32 loopCnt = _min(numFaces, NUM_FACE_ITERS);
                    const Face* faces = ct.faces;
                    const FaceCluster* fclusters = ct.faceClusters;
                    for ( u32 k=0; k<loopCnt; ++k )
                    {
                        const Face* face = faces + node.getFace(fclusters, k);
                        float u, v;
                        float dist = FaceRayIntersect(face, pb.point, ray.d, ct.meshDataPtrs, u, v);
                        if ( dist < result.dist )
                        {
                            result.u = u;
                            result.v = v;
                            result.dist = dist;
                            result.face = face;
                            result.ray  = pb.ray;
                        }
                    }
                }

                float distToBox = FLT_MAX;
                u32 nextBoxId   = SelectNextBox(&node.bMin, ct.sides + node.right*6, ray.sign, pb.point, ray.invd, distToBox);
                if ( nextBoxId != RL_INVALID_INDEX && result.dist > distToBox )
                {
                    pb.point += ray.d*(distToBox+MARCH_EPSILON);
                    pb.node   = nextBoxId;
                }
                else
                {
                    break;
                }
            }
            else
            {
                // both must be valid
                assert(RL_VALID_INDEX(node.left) && RL_VALID_INDEX(BVH_GET_INDEX(node.right)));
                u32 spAxis  = BVH_GET_AXIS(node.right);
                float s = (node.bMax[spAxis] + node.bMin[spAxis])*.5f;
                if ( pb.point[spAxis] < s ) { pb.node = node.left; }
                else { pb.node = BVH_GET_INDEX(node.right); }
            }
        }

        ct.hitResults[pb.localId] = result;
    }

    GLOBAL void SetupRays(u32 numRays, u32 tileOffset, Store<PointBox>* pbIn)
    {
        u32 localId = bIdx.x * bDim.x + tIdx.x;
        if ( localId >= numRays ) return;
        FirstRays( tileOffset + localId, localId, pbIn );
    }

    GLOBAL void ResetHitResults(u32 numRays, HitResult* hitResults)
    {
        u32 localId = bIdx.x * bDim.x + tIdx.x;
        if ( localId >= numRays ) return;
        SHARED HitResult sResults[RL_BLOCK_THREADS];
        sResults[tIdx.x] = hitResults[localId];
        sResults[tIdx.x].dist = FLT_MAX;
        hitResults[localId] = sResults[tIdx.x];
        //hitResults[localId].dist = FLT_MAX;
        //ct.hitResults[localId].dist = FLT_MAX;
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
    GLOBAL void UpdateQueueMappingKernel(u32 queueLength, QIn* qIn, byte* id2Queue)
    {
        u32 localId = bIdx.x * bDim.x + tIdx.x;
        if ( localId >= queueLength ) return;
    #if _DEBUG
        bool inserted=false;
    #endif
        u32 nextTop=0;
        for ( u32 i=0; i<RL_NUM_INNER_QUEUES; i++ )
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
    DEVICE_DYN u32 UpdateQueueMapping(QIn& qIn, byte* id2Queue)
    {
        u32 tops[RL_NUM_INNER_QUEUES];
        u32 qlength = GetQueueTop(qIn, tops);
        if ( qlength==0 ) return qlength;
    #if RL_USE_INNER_QUEUES
        u32 offsets[RL_NUM_INNER_QUEUES];
        u32 total=0;
        for ( u32 i=0; i<RL_NUM_INNER_QUEUES; ++i )
        {
            offsets[i] = total;
            total += tops[i];
        }
        u32 err = hostOrDeviceCpy(((char*)qIn) + offsetof(Store<PointBox>, m_offsets[0]), offsets, 4*RL_NUM_INNER_QUEUES, cudaMemcpyHostToDevice, false); // cannot do async as offsets is on stack
        assert(err==0);
        dim3 blocks  ((qlength + RL_BLOCK_THREADS-1)/RL_BLOCK_THREADS);
        dim3 threads (RL_BLOCK_THREADS);
        RL_KERNEL_CALL(RL_BLOCK_THREADS, blocks, threads, UpdateQueueMappingKernel, qlength, qIn, id2Queue);
        dynamicSync();
    #endif
        return qlength;
    }

    template <class QIn>
    DEVICE_DYN u32 GetQueueTop(QIn& dQueue, u32* tops)
    {
    #if RL_CUDA_DYN
        return dQueue->getTotalLength();
    #else
        u32 queueLength = 0;
    #if !RL_USE_INNER_QUEUES
        u32 err = hostOrDeviceCpy(&queueLength, ((char*)dQueue) + offsetof(Store<PointBox>, m_top), 4, cudaMemcpyDeviceToHost, false);
        assert(err==0);
    #else
        u32 err = hostOrDeviceCpy(tops, ((char*)dQueue) + offsetof(Store<PointBox>, m_top), 4*RL_NUM_INNER_QUEUES, cudaMemcpyDeviceToHost, false);
        assert(err==0);
        for ( u32 i=0; i< RL_NUM_INNER_QUEUES; ++i ) queueLength += tops[i];
    #endif
        return queueLength;
    #endif
    }

    template <class QIn>
    DEVICE_DYN void SetQueueTop(QIn& dQueue, u32 val=0)
    {
    #if RL_CUDA_DYN
        dQueue->resetTop();
    #else
    #if !RL_USE_INNER_QUEUES
        u32 err = hostOrDeviceCpy(((char*)dQueue) + offsetof(Store<PointBox>, m_top), &val, 4, cudaMemcpyHostToDevice, false); // cannot do async as val is on stack
        assert(err==0);
    #else
        assert(val < 256);
        u32 tops[RL_NUM_INNER_QUEUES];
        for ( auto& t : tops ) t=val;
        u32 err = hostOrDeviceCpy(((char*)dQueue) + offsetof(Store<PointBox>, m_top), tops, 4*RL_NUM_INNER_QUEUES, cudaMemcpyHostToDevice, false); // cannot do async as tops is on stack
        assert(err==0);
    #endif
    #endif
    }

    template <class QIn, class Qout, class Func>
    DEVICE_DYN void DoQueries(const char* queriesName, u32 queueLength, u32 quotumThreshold, 
                              Store<QIn>*& queueIn, Store<QIn>*& queueOut, Store<Qout>* remainderQueue, Func f)
    {
        assert( queueIn != queueOut );

        DBG_QUERIES_BEGIN(queriesName);

        // if no queries left to solve, quit
        u32 iterNum=0;
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

            queueLength = UpdateQueueMapping( queueOut, h_ct.id2Queue );
            DBG_QUERIES_OUT(queueLength);

            // Swap queues
            auto* qTemp = queueIn;
            queueIn  = queueOut;
            queueOut = qTemp;
            assert( queueIn != queueOut );
            ++iterNum;
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

        u32 queueLength = UpdateQueueMapping( h_ct.pbQueueIn, h_ct.id2Queue );
        while ( queueLength > POINTBOX_STOP_QUOTEM && h_ct.curDepth < h_ct.maxDepth )
        {
            blocks.x = ((numRays + RL_BLOCK_THREADS-1)/RL_BLOCK_THREADS);
            RL_KERNEL_CALL( RL_BLOCK_THREADS, blocks, threads, ResetHitResults, numRays, h_ct.hitResults );
            dynamicSync();

            blocks.x = ((queueLength + RL_BLOCK_THREADS-1)/RL_BLOCK_THREADS);
            RL_KERNEL_CALL(RL_BLOCK_THREADS, blocks, threads, SolveRay, queueLength, h_ct.pbQueueIn);
            dynamicSync();

            // Solve all intersections iteratively
            //u32 numIters = 0;
            //while ( queueLength > POINTBOX_STOP_QUOTEM )
            //{
            //    // Iterate Point-box queries until queues are empty
            //    DoQueries("Point-box", queueLength, POINTBOX_STOP_QUOTEM, h_ct.pbQueueIn, h_ct.pbQueueOut, h_ct.leafQueueIn, PointBoxKernel);

            //    // Iterate Ray/leaf queries until queues are empty
            //    queueLength = UpdateQueueMapping( h_ct.leafQueueIn, h_ct.id2Queue );
            //    DoQueries("Ray-leaf", queueLength, RAYLEAF_STOP_QUOTEM, h_ct.leafQueueIn, h_ct.leafQueueOut, h_ct.pbQueueIn, RayLeafKernel );

            //    queueLength = UpdateQueueMapping( h_ct.pbQueueIn, h_ct.id2Queue );
            //    DBG_QUERIES_RESTART( queueLength );
            //    numIters++;
            //} // End while there are still point-box queries

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
        vec3 dir = TD.orient * TD.rayDirs[globalId];
        vec3 ori = TD.eye; /*vec3(0, 0, -2.5f);*/
        QueueRay(ori, dir, pbIn);
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