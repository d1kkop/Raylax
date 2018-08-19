#pragma once
#include "Reylax_internal.h"
#include <vector>


#define RL_INVALID_INDEX ((u32)-1)
#define RL_VALID_INDEX( idx ) (idx!=RL_INVALID_INDEX)

#define BVH_MAX_DEPTH 64
#define BVH_NUM_FACES_IN_LEAF 32
#define BVH_DBG_INFO 0

#define BVH_ISLEAF( idx ) (((idx)>>31)==1)
#define BVH_GETNUM_TRIANGLES(idx) ((idx)&0x7FFFFFF)
#define BVH_GET_INDEX(idx) ((idx)&0x3FFFFFF)
#define BVH_SET_INDEX(idx, v) ((idx)|=((v)&0x3FFFFFF))
#define BVH_SET_LEAF_AND_FACES(idx, kFaces) ((idx)=(1<<31)|(kFaces))
#define BVH_SET_AXIS(idx, axis) ((idx)|=((axis)<<30))
#define BVH_GET_AXIS(idx) ((idx)>>30)



namespace Reylax
{
     struct DeviceBuffer;

     __align__(4)
     struct Ray
     {
         vec3 o, d, invd;
         char sign[3];
     };

     __align__(4)
     struct PointBox
     {
         vec3 point;
         u32 node;
         u32 localId;
         u32 ray;
     };

     __align__(4)
     struct RayLeaf
     {
         u32 localId;
         u32 node;
         u32 faceIdx;
         u32 ray;
     };

     __align__(4)
     struct RayFace
     {
         u32 ray;
         u32 face;
     };

     // Each leaf BVH node has a faceCluster. It holds a list of faces.
     __align__(4)
    struct FaceCluster
    {
        u32 faces[BVH_NUM_FACES_IN_LEAF];
        u32 numFaces;
        FDEVICE INLINE u32 getFace(u32 idx) const
        {
            assert(idx < BVH_NUM_FACES_IN_LEAF && idx < numFaces);
            return faces[idx];
        }
    };

    struct BvhNode
    {
        vec3 bMin, bMax;
        u32 left, right;

        FDEVICE bool isLeaf() const
        {
            return BVH_ISLEAF(left);
        }

        FDEVICE u32 numFaces() const
        {
            assert(isLeaf());
            return BVH_GETNUM_TRIANGLES(left);
        }

        FDEVICE u32 getFace(const FaceCluster* faceClusters, u32 idx) const
        {
            assert(isLeaf());
            const FaceCluster* fc = faceClusters + right; // No split axis stored in right in case of leaf. Safe to to use without GET_INDEX.
        #if _DEBUG
            u32 tnumFaces=numFaces();
            assert(tnumFaces==fc->numFaces);
        #endif
            return fc->getFace(idx);
        }

        static u32 build( const MeshData** meshData, u32 numMeshDatas, 
                          DeviceBuffer** ppBvhTree,
                          DeviceBuffer** ppFaces,
                          DeviceBuffer** ppFaceClusters,
                          DeviceBuffer** ppSides,
                          vec3& worldMin,
                          vec3& worldMax );

        static void determineCentre(std::vector<Face>& faces, const MeshData** meshData, vec3& centre);
        static void determineBbox(std::vector<Face>& faces, const MeshData** meshData, vec3& bMin, vec3& bMax);

        static void showDebugInfo( const BvhNode* nodes );
    };

    template <typename T>
    struct Store
    {
        T*  m_elements;
        u32 m_top;
        u32 m_max;

        FDEVICE T* getNew(u32 cnt=1)
        {
            u32 old = atomicAdd2<u32>(&m_top, cnt);
            assert(validate(old+cnt));
            memset(m_elements + old, 0, sizeof(T)*cnt);
            return m_elements + old;
        }

        FDEVICE T* get(u32 idx) const
        {
            assert(idx < m_top);
            return m_elements + idx;
        }

        FDEVICE bool validate(u32 newTop) const
        {
            if ( newTop > m_max ) printf("m_top = %d, _max = %d\n", newTop, m_max);
            return newTop <= m_max;
        }
    };

    // Tracer context
    __align__(8)
    struct TracerContext
    {
        vec3 bMin, bMax;
        Store<Ray>*      rayPayload;
        Store<PointBox>* pbQueues[2];
        Store<RayLeaf>*  leafQueues[2];
        const BvhNode* bvhNodes;
        const Face* faces;
        const FaceCluster* faceClusters;
        const u32* sides;
        const MeshData* const* meshData;
        // -- Changes every kernel run 0, to 1 ---
        u32 queueIn, queueOut;
        // Output
        HitResult* hitResults;
        // Callbacks
        RaySetupFptr setupCb;
        HitResultFptr   hitCb;
        // constants
        u32 curDepth;
        u32 maxDepth;
    };


    FDEVICE INLINE float TriIntersect(const vec3 &orig, const vec3 &dir, const vec3 &v0, const vec3 &v1, const vec3 &v2, float& u, float& v)
    {
        vec3 v0v1 = v1 - v0;
        vec3 v0v2 = v2 - v0;
        vec3 pvec = cross(dir, v0v2);
        float det = dot(v0v1, pvec);

        //#ifdef CULLING 
        //    // if the determinant is negative the triangle is backfacing
        //    // if the determinant is close to 0, the ray misses the triangle
        //    if (det < kEpsilon) return FLT_MAX;
        //#else 
        //    // ray and triangle are parallel if det is close to 0
        //    if (fabs(det) < kEpsilon) return FLT_MAX; 
        //#endif 
        //    float invDet = 1.f / det; 
        // 

            // IF determinate is small or zero, invDet will be large or inifnity, in either case the the computations remain valid.

        float invDet = 1.f/det;
        vec3 tvec = orig - v0;
        vec3 qvec = cross(tvec, v0v1);
        u = dot(tvec, pvec) * invDet;
        v = dot(dir, qvec) * invDet;
        //   float dist = dot(v0v2, qvec) * invDet;

        return  (u<0||u>1 ? FLT_MAX :
                (v<0||v+u>1 ? FLT_MAX :
                (dot(v0v2, qvec)*invDet)));
    }

    FDEVICE INLINE float FaceRayIntersect(const Face* face, const vec3& eye, const vec3& dir, const MeshData* const* meshData, float& u, float& v)
    {
        assert(meshData);
        const MeshData* mesh = meshData[face->w];
        const vec3* vp = (vec3*)mesh->vertexData[VERTEX_DATA_POSITION];
        assert(mesh->vertexDataSizes[VERTEX_DATA_POSITION] == 3);
        return TriIntersect(eye, dir, vp[face->x], vp[face->y], vp[face->z], u, v);
    }

    // https://tavianator.com/fast-branchless-raybounding-box-intersections/
    FDEVICE INLINE float BoxRayIntersect(const vec3& bMin, const vec3& bMax, const vec3& orig, const vec3& invDir)
    {
        vec3 tMin  = (bMin - orig) * invDir;
        vec3 tMax  = (bMax - orig) * invDir;
        vec3 oMin  = _min(tMin, tMax);
        vec3 oMax  = _max(tMin, tMax);
        float dmin = _max(oMin.x,_max(oMin.y, oMin.z));
        float dmax = _min(oMax.x,_min(oMax.y, oMax.z));
        float dist = _max(0.f, dmin);
        return (dmax >= dmin ? dist : FLT_MAX);
    }

    FDEVICE INLINE u32 SelectNextBox(const vec3* bounds, const u32* links, const char* sign, 
                                     const vec3& p, const vec3& rinvd, float& tOut)
    {
        float xDist = (bounds[sign[0]].x - p.x) * rinvd.x;
        float yDist = (bounds[sign[1]].y - p.y) * rinvd.y;
        float zDist = (bounds[sign[2]].z - p.z) * rinvd.z;

        // assume xDist being the smallest
        u32 offset = 0;
        u32 side   = 0;

        bool bEval;

        // check if yDist > xDist
        bEval  = yDist < xDist;
        if ( bEval )
        {
            xDist  = yDist;
            offset = 2;
            side = 1;
        }
        //xDist  = bEval? yDist : xDist;
        //offset = bEval? 2 : 0;
        //side   = bEval? 1 : 0;

        // check if zDist < xDist, note: xDist was updated if yDist was smaller
        bEval  = zDist < xDist;
        if ( bEval )
        {
            tOut = zDist;
            offset = 4;
            side = 2;
        } else tOut = xDist;
        //tOut   = bEval? zDist : xDist;
        //offset = bEval? 4 : offset;
        //side   = bEval? 2 : side;

        return links[offset + sign[side]];
    }

    FDEVICE INLINE bool AABBOverlap(const vec3& tMin, const vec3& tMax, const vec3& bMin, const vec3& bMax)
    {
        bool b =
            (tMin[0] > bMax[0] ? false :
            (tMin[1] > bMax[1] ? false :
             (tMin[2] > bMax[2] ? false :
             (tMax[0] < bMin[0] ? false :
              (tMax[1] < bMin[1] ? false :
              (tMax[2] < bMin[2] ? false : true))))));
        return b;
    }

    FDEVICE INLINE bool AABBOverlap2(const vec3& tMin, const vec3& tMax, const vec3& bMin, const vec3& bMax)
    {
        if ( tMin[0] > bMax[0] ) return false;
        if ( tMin[1] > bMax[1] ) return false;
        if ( tMin[2] > bMax[2] ) return false;
        if ( tMax[0] < bMin[0] ) return false;
        if ( tMax[1] < bMin[1] ) return false;
        if ( tMax[2] < bMin[2] ) return false;
        return true;
    }

    FDEVICE INLINE bool PointInAABB(const vec3& pnt, const vec3& bMin, const vec3& bMax)
    {
        if ( pnt[0] > bMax[0] ) return false;
        if ( pnt[1] > bMax[1] ) return false;
        if ( pnt[2] > bMax[2] ) return false;
        if ( pnt[0] < bMin[0] ) return false;
        if ( pnt[1] < bMin[1] ) return false;
        if ( pnt[2] < bMin[2] ) return false;
        return true;
    }

    FDEVICE INLINE void ValidateAABB(const vec3& bMin, const vec3& bMax)
    {
        u32 dZero = 0;
        dZero += (bMax[0]-bMin[0]<0) ? 1 : 0;
        dZero += (bMax[1]-bMin[1]<0) ? 1 : 0;
        dZero += (bMax[2]-bMin[2]<0) ? 1 : 0;
        if ( dZero==3 )
        {
            printf("BoxSize: %.f %.f %.f\nbMin %f %f %f | bMax %f %f %f\n",
                   bMax[0]-bMin[0], bMax[1]-bMin[1], bMax[2]-bMin[2],
                   bMin[0], bMin[1], bMin[2], bMax[0], bMax[1], bMax[2]);
            assert(false);
        }
    }

    FDEVICE INLINE void PrintAABB(const vec3& bMin, const vec3& bMax)
    {
        printf("BoxSize: %.f %.f %.f\nbMin %f %f %f | bMax %f %f %f\n",
               bMax[0]-bMin[0], bMax[1]-bMin[1], bMax[2]-bMin[2],
               bMin[0], bMin[1], bMin[2], bMax[0], bMax[1], bMax[2]);
    }
}
