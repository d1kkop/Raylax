#include "ReylaxCommon.h"

#define MAX_HITS_PER_RAY 16
#define NUM_RAYBOX_QUEUES 32
#define NUM_LEAF_QUEUES 32


namespace Reylax
{
    struct Ray
    {
        vec3 o, d;
    };

    struct RayBox
    {
        u32 ray;
        u32 node;
    };

    struct RayFace
    {
        u32 ray;
        u32 face;
    };

    struct RayFaceHitCluster
    {
        RayFaceHitResult results[MAX_HITS_PER_RAY];
        u32 count;
    };


    GLOBAL void RayBoxKernel(u32 numRayBoxes,
                             Store<RayBox>* rayBoxQueueIn,
                             Store<RayBox>* rayBoxQueueOut,
                             Store<u32>* leafQueue,
                             Ray* rayBuffer,
                             BvhNode* bvhNodes)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= numRayBoxes ) return;

        RayBox* rb    = rayBoxQueueIn->get(i);
        Ray* ray      = rayBuffer + rb->ray;
        BvhNode* node = bvhNodes + rb->node;

        vec3 o   = ray->o;
        vec3 d   = ray->d;
        vec3 cp  = node->cp;
        vec3 hs  = node->hs;
        vec3 invDir(1.f/d.x, 1.f/d.y, 1.f/d.z);

        if ( BoxRayIntersect(cp-hs, cp+hs, o, invDir) )
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
                                       BvhNode* bvhNodes)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= numInnerQueues ) return;
    }

    GLOBAL void LeafExpandKernel(u32 numLeafs,
                                 Store<RayBox>* rayBoxQueue,
                                 Store<u32>* leafQueue,
                                 Store<RayFace>* rayFaceQueue,
                                 BvhNode* bvhNodes,
                                 FaceCluster* faceClusters)
    {
        u32 i = bIdx.x * bDim.x + tIdx.x;
        if ( i >= numLeafs ) return;
        u32 rayBoxIdx = *leafQueue->get(i);
        RayBox* rb    = rayBoxQueue->get(rayBoxIdx);
        BvhNode* node = bvhNodes + rb->node;
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
            assert( curHitIdx < MAX_HITS_PER_RAY );
            if ( curHitIdx < MAX_HITS_PER_RAY )
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
void bmMarchProgressive(void* rays, u32 numRays,
                        void* rayBoxQueue,
                        void* leafQueue,
                        void* rayFaceQueue,
                        void* bvhNodes,
                        void* faceClusters,
                        void* hitResultClusters,
                        void* meshDataPtrs)
{



#if CUDA

    //bmFindClosestHit<<< 1, 1 >>> (
    //    nullptr, 
    //    0, 
    //    nullptr,
    //    nullptr,
    //    &bmShadeNormal
    //    );

#else

#endif
}


extern "C"
{

}