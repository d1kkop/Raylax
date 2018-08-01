#include "ReylaxCommon.h"
#include "Reylax.h"
using namespace std;

extern "C"
int rlTriBoxOverlap(const float* boxcenter, const float* boxhalfsize, const vec3 triverts[3]);


namespace Reylax
{
    u32 BvhNode::build(const MeshData* meshData, u32 numMeshDatas)
    {
        if ( !meshData || numMeshDatas == 0 ) return ERROR_INVALID_PARAMETER;

        vector<Face> faces;

        for ( u32 i=0; i < numMeshDatas; ++i )
        {
            const MeshData* md = meshData + i;
            for ( u32 j=0; j<md->numIndices; j+= 3 )
            {
                const u32 id0  = md->indices[j];
                const u32 id1  = md->indices[j+1];
                const u32 id2  = md->indices[j+2];
                Face f;
                f.x = id0;
                f.y = id1;
                f.z = id2;
                f.w = i;
                faces.push_back( f );
            }
        }

        BvhNode* nodes = new BvhNode[1024*1024];
        u32 nodeCount =  step( nodes, 0, 0, meshData, faces );


        return ERROR_ALL_FINE;
    }


    u32 BvhNode::step(BvhNode* nodes, u32 nodeIdx, u32 depth, const MeshData* meshData, vector<Face> facesCopy)
    {
        if ( depth == BVH_MAX_DEPTH || (u32)facesCopy.size() <= BVH_NUM_FACES_IN_LEAF )
        {
            if ( (u32)facesCopy.size() > BVH_NUM_FACES_IN_LEAF )
            {
                printf("Could not fit all faces in a leaf node of BVH while max depth was reached.\n");
            }
            // TODO insert faces
            return nodeIdx;
        }

        BvhNode* node = nodes + nodeIdx;
        vector<Face> intersectingFaces;
        for ( auto& f : facesCopy )
        {
            const vec3* vd  = (const vec3*)(meshData + f.w)->vertexData[VERTEX_DATA_POSITION];
            const vec3 v[3] = { vd[f.x], vd[f.y], vd[f.z] };
            if ( rlTriBoxOverlap(&node->cp.x, &node->hs.x, v) == 1 )
            {
                intersectingFaces.push_back( f );
            }
        }

        node->left = nodeIdx+1;
        nodeIdx = step( nodes, node->left, depth + 1, meshData, facesCopy );
        node->right = nodeIdx+1;
        nodeIdx = step( nodes, node->right, depth + 1, meshData, facesCopy );

        // return count;
        return nodeIdx + 1;
    }

}