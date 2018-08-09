#include "ReylaxCommon.h"
#include "Reylax.h"
#include "Reylax_internal.h"

#define NUM_RAYBOX_QUEUES 32
#define NUM_LEAF_QUEUES 32

#define RAY_BOX_THREADS 256


namespace Reylax
{
    GLOBAL void RayBoxKernel(vec3 eye,
                             mat3 orient,
                             u32 numRayBoxes,
                             Store<RayBox>* rayBoxQueueIn,
                             Store<RayBox>* rayBoxQueueOut,
                             Store<u32>* leafQueue,
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
                u32* leafTestIdx = leafQueue->getNew(1);
                *leafTestIdx = i;
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
                                       Store<u32>* leafQueue,
                                       Ray* rayBuffer,
                                       const BvhNode* bvhNodes)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= numInnerQueues ) return;
    }

    GLOBAL void LeafExpandKernel(u32 numLeafs,
                                 Store<RayBox>* rayBoxQueue,
                                 Store<u32>* leafQueue,
                                 Store<RayFace>* rayFaceQueue,
                                 const BvhNode* bvhNodes,
                                 FaceCluster* faceClusters)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= numLeafs ) return;
        u32 rayBoxIdx = *leafQueue->get(i);
        RayBox* rb    = rayBoxQueue->get(rayBoxIdx);
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

    u32 rlRayBox(const float* eye3,
                 const float* orient3x3,
                 u32 numRayBoxes,
                 Store<RayBox>* rayBoxQueueIn,
                 Store<RayBox>* rayBoxQueueOut,
                 Store<u32>* leafQueue,
                 const vec3* rayDirs,
                 const BvhNode* bvhNodes)
    {
        if ( numRayBoxes==0 || !rayBoxQueueIn || !rayBoxQueueOut || !leafQueue || !rayDirs || !bvhNodes )
        {
            return ERROR_INVALID_PARAMETER;
        }

        dim3 blocks  ((numRayBoxes + RAY_BOX_THREADS-1)/RAY_BOX_THREADS);
        dim3 threads (RAY_BOX_THREADS);
        vec3 eye = *(vec3*)eye3;
        mat3 orient = *(mat3*)orient3x3;
    #if RL_CUDA
        RayBoxKernel<<< blocks, threads >>>( eye, orient, numRayBoxes, rayBoxQueueIn, rayBoxQueueOut, leafQueue, rayDirs, bvhNodes );
    #else
        emulateCpu(RAY_BOX_THREADS, blocks, threads, [=]()
        { 
            RayBoxKernel( eye, orient, numRayBoxes, rayBoxQueueIn, rayBoxQueueOut, leafQueue, rayDirs, bvhNodes );
        });
    #endif

        return ERROR_ALL_FINE;
    }
}