#include "ReylaxCommon.h"
#include "Reylax_internal.h"
#include "Reylax.h"
#include "DeviceBuffer.h"
using namespace std;


int rlTriBoxOverlap(const float* boxcenter, const float* boxhalfsize, const vec3 triverts[3]);


namespace Reylax
{
    void BvhNode::splitDefault(const MeshData* const * meshPtrs, const std::vector<Face>& faces, const vec3& bMin, const vec3& bMax, float& s, u32& splitAxis)
    {
        vec3 hs = (bMax-bMin)*.5f;
        float biggest = hs.x;
        splitAxis = 0;
        if ( hs.y > biggest ) { splitAxis = 1; biggest = hs.y; }
        if ( hs.z > biggest ) { splitAxis = 2; }
        s = (bMax[splitAxis]+bMin[splitAxis])*.5f;
    }

    void BvhNode::splitCenter(const MeshData* const * meshPtrs, const std::vector<Face>& faces, const vec3& bMin, const vec3& bMax, float& s, u32& splitAxis)
    {
        double cp[3]{};
        u32 numAdds=0;
        for ( auto& f : faces )
        {
            auto vd = (const vec3*) meshPtrs[f.w]->vertexData[VERTEX_DATA_POSITION];
            const vec3 v[] = { vd[f.x], vd[f.y], vd[f.z] };
            for ( auto& vv : v )
            {
                if ( PointInAABB(vv, bMin, bMax) )
                {
                    cp[0] += vv.x;
                    cp[1] += vv.y;
                    cp[2] += vv.z;
                    numAdds++;
                }
            }
        }
        if ( numAdds==0 )
        {
            splitDefault( meshPtrs, faces, bMin, bMax, s, splitAxis );
            return;
        }
        cp[0] /= numAdds;
        cp[1] /= numAdds;
        cp[2] /= numAdds;
        // -- 
        vec3 hs = (bMax-bMin)*.5f;
        float biggest = hs.x;
        splitAxis = 0;
        if ( hs.y > biggest ) { splitAxis = 1; biggest = hs.y; }
        if ( hs.z > biggest ) { splitAxis = 2; }
        s = (float)cp[splitAxis];
        float t = (bMax[splitAxis]-bMin[splitAxis]);
        float failC = 0.02f;
        if ( (s-bMin[splitAxis])/t < failC || (bMax[splitAxis]-s)/t < failC )
        {
            splitDefault( meshPtrs, faces, bMin, bMax, s, splitAxis );
        }
    }

    void BvhNode::splitMedian(const MeshData* const * meshPtrs, const std::vector<Face>& faces, const vec3& bMin, const vec3& bMax, float& s, u32& splitAxis)
    {
    }
}