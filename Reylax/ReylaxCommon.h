#pragma once
#include "ReylaxCuda.h"


namespace Reylax
{
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

        FDEVICE T* get(u32 idx)
        {
            assert(idx < m_top);
            return m_elements + idx;
        }

        FDEVICE bool validate(u32 newTop)
        {
            if ( newTop > m_max ) printf("m_top = %d, _max = %d\n", newTop, m_max);
            return newTop <= m_max;
        }
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

    FDEVICE INLINE float FaceRayIntersect(Face* face, const vec3& eye, const vec3& dir, const MeshData* meshData, float& u, float& v)
    {
        assert(meshData);
        const MeshData* mesh = &meshData[face->w];
        vec3* vp = (vec3*)mesh->vertexData[VERTEX_DATA_POSITION];
        assert(mesh->vertexDataSizes[VERTEX_DATA_POSITION] == 3);
        return TriIntersect(eye, dir, vp[face->x], vp[face->y], vp[face->z], u, v);
    }

    // https://tavianator.com/fast-branchless-raybounding-box-intersections/
    FDEVICE INLINE float BoxRayIntersect(const vec3& bMin, const vec3& bMax, const vec3& orig, const vec3& invDir)
    {
        vec3 tMin   = (bMin - orig) * invDir;
        vec3 tMax   = (bMax - orig) * invDir;
        vec3 tMax2  = _max(tMin, tMax);
        float ftmax = _min(tMax2.x, _min(tMax2.y, tMax2.z));
        if ( ftmax < 0.f ) return FLT_MAX;
        vec3 tMin2  = _min(tMin, tMax);
        float ftmin = _max(tMin2.x, _max(tMin2.y, tMin2.z));
        float dist  = _max(0.f, ftmin);
        dist = (ftmax >= ftmin ? dist : FLT_MAX);
        return dist;
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
