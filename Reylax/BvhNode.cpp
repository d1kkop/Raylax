#include "ReylaxCommon.h"
#include "Reylax_internal.h"
#include "Reylax.h"
#include "DeviceBuffer.h"
using namespace std;


int rlTriBoxOverlap(const float* boxcenter, const float* boxhalfsize, const vec3 triverts[3]);


namespace Reylax
{
    // -------- BvhNode -------------------------------------------------------------------------------------------------

    u32 BvhNode::build(const MeshData** meshData, u32 numMeshDatas,
                       DeviceBuffer** ppBvhTree,
                       DeviceBuffer** ppFaces,
                       DeviceBuffer** ppFaceClusters)
    {
        if ( !meshData || numMeshDatas == 0 || !ppBvhTree || !ppFaces || !ppFaceClusters )
        {
            return ERROR_INVALID_PARAMETER;
        }

        // create stack
        struct stNode
        {
            u32 parentIdx, depth;
            vector<Face> faces;
            vec3 cp, hs;
        };

        const u32 stSize = 64;
        stNode stack[stSize];

        // find total face count and set min max bbox
        u32 allocatedFaceCount = 0;
        vec3 bMin(FLT_MAX);
        vec3 bMax(FLT_MIN);
        forEachFace(meshData, numMeshDatas, [&](u32 mId, const u32 id[3], const vec3 v[3])
        {
            for ( u32 i=0; i<3; i++ )
            {
                const vec3& vi = v[i];
                bMin = _min(bMin, vi);
                bMax = _max(bMax, vi);
            }
            Face f;
            f.x = id[0];
            f.y = id[1];
            f.z = id[2];
            f.w = mId;
            f.mat = nullptr;
            stack[0].faces.push_back( f );
            allocatedFaceCount++;
        });
        allocatedFaceCount *= 200;

        // enlarge bbox slightly
        vec3 dBox = (bMax-bMin)*0.00001f;
        bMin -= dBox;
        bMax += dBox;

        // allocate sufficient amount of data
        u32 numNodes = 1024*1024*8;
        u32 numFacesClusters = _max<u32>((u32)sqrt(numNodes), allocatedFaceCount/(BVH_NUM_FACES_IN_LEAF/8));// (u32)sqrt(numNodes);

    #if RL_PRINT_STATS
        float bvhSize  = (float)sizeof(BvhNode)*numNodes/1024/1024;
        float faceSize = (float)sizeof(Face)*allocatedFaceCount/1024/1024;
        float fclusSize= (float)sizeof(BvhNode)*numFacesClusters/1024/1024;
        printf("Intermediate data allocation\n");
        printf("BvhNodes: %d, size %.3fmb\n", numNodes, bvhSize);
        printf("Faces: %d, size %.3fmb\n", allocatedFaceCount, faceSize);
        printf("Fclusters: %d, size %.3fmb\n", numFacesClusters, fclusSize);
        printf("Total: %.3fmb\n", (bvhSize+faceSize+fclusSize));
    #endif

        BvhNode* nodes  = new BvhNode[numNodes];
        Face* faces     = new Face[allocatedFaceCount]; // times 2 to accomodate for overlapping faces on two boxes
        FaceCluster* fc = new FaceCluster[numFacesClusters];

        // indexers into memory buffers
        u32 nodeIndexer=0;
        u32 faceIndexer = 0;
        u32 faceClusterIndexer = 0;
        
        // setup remaining part of first node in stack (faces for stack[0] have been poplated already)
        stNode* st = stack;
        st->parentIdx = RL_INVALID_INDEX;
        st->depth = 0;
        st->cp = (bMax+bMin)*.5f;
        st->hs = (bMax-bMin)*.5f;
        i32 top=0;

        // generate tree
        while ( top >= 0 )
        {
            st = &stack[top--];

            // If cur stack node has valid parent index, we can now efficiently assign index children of parent
            // such that the indexing of nodes: 0, 1, 2... is linear in memory.
            if ( RL_VALID_INDEX(st->parentIdx) )
            {
                BvhNode* parent = nodes + st->parentIdx;
                if ( !RL_VALID_INDEX(parent->left) ) parent->left = nodeIndexer;
                else
                {
                    assert(!RL_VALID_INDEX(parent->right));
                    parent->right = nodeIndexer;
                }
            }
            BvhNode* node = nodes + nodeIndexer++;
            node->cp = st->cp;
            node->hs = st->hs;
            node->left  = RL_INVALID_INDEX;
            node->right = RL_INVALID_INDEX;

            if ( st->depth == BVH_MAX_DEPTH || (u32)st->faces.size() <= BVH_NUM_FACES_IN_LEAF )
            {
                if ( (u32)st->faces.size() > BVH_NUM_FACES_IN_LEAF )
                {
                    printf("Could not fit all faces in a leaf node of BVH while max depth was reached.\n");
                }

                assert( faceClusterIndexer < numFacesClusters );
                node->left  = (1<<31) | (u32)st->faces.size(); // Leaf bit | num triangles
                node->right = faceClusterIndexer++;
                FaceCluster* cluster = fc + node->right;
                cluster->numFaces = _min( (u32)st->faces.size(), (u32)BVH_NUM_FACES_IN_LEAF );
                for ( u32 i=0; i<cluster->numFaces; ++i )
                {
                    assert( faceIndexer < allocatedFaceCount );
                    Face* fNew = faces + faceIndexer++;
                    *fNew = st->faces[0];
                    cluster->faces[i] = faceIndexer-1;
                }
            }
            else
            {
                // determine centre
                vec3 centre(0);
                for ( auto& f : st->faces )
                {
                    const vec3* vd  = (const vec3*)(meshData[f.w])->vertexData[VERTEX_DATA_POSITION];
                    const vec3 v[3] ={ vd[f.x], vd[f.y], vd[f.z] };
                    for ( auto& vi: v ) centre += vi;
                }
                centre /= (float)st->faces.size();

                u32 splitAxis = 0;
                float biggest = node->hs.x;
                if ( node->hs.y > biggest ) { splitAxis = 1; biggest = node->hs.y; }
                if ( node->hs.z > biggest ) { splitAxis = 2; }

                bMin = st->cp-st->hs;
                bMax = st->cp+st->hs;
                vec3 lMax = bMax;
                vec3 rMin = bMin;
                lMax[splitAxis] = centre[splitAxis];
                rMin[splitAxis] = centre[splitAxis];

                vec3 cpL = (bMin+lMax)*.5f;
                vec3 hsL = (lMax-bMin)*.5f;
                vec3 cpR = (rMin+bMax)*.5f;
                vec3 hsR = (bMax-rMin)*.5f;

                vector<Face> facesL, facesR;
                for ( auto& f : st->faces )
                {
                    const vec3* vd  = (const vec3*)(meshData[f.w])->vertexData[VERTEX_DATA_POSITION];
                    const vec3 v[3] = { vd[f.x], vd[f.y], vd[f.z] };
                    if ( rlTriBoxOverlap(&cpL.x, &hsL.x, v) == 1 ) facesL.push_back(f);
                    if ( rlTriBoxOverlap(&cpR.x, &hsR.x, v) == 1 ) facesR.push_back(f);
                }

            #if RL_PRINT_STATS
                printf("Faces left: %zd\n", facesL.size());
                printf("Faces right: %zd\n", facesR.size());
            #endif

                u32 depth = st->depth;
                assert(top+2 < stSize);

                // right
                st = &stack[++top];
                st->cp = cpR;
                st->hs = hsR;
                st->depth = depth+1;
                st->parentIdx = nodeIndexer-1;
                st->faces = move(facesL);
                // left
                st = &stack[++top];
                st->cp = cpL;
                st->hs = hsL;
                st->depth = depth+1;
                st->parentIdx = nodeIndexer-1;
                st->faces = move(facesR);
            } // split
        } // while

        // setup device memory
        *ppBvhTree = new DeviceBuffer( sizeof(BvhNode) * nodeIndexer );
        *ppFaces   = new DeviceBuffer( sizeof(Face) * faceIndexer );
        *ppFaceClusters = new DeviceBuffer( sizeof(FaceCluster) * faceClusterIndexer );

        (*ppBvhTree)->copyFrom( nodes, false );
        (*ppFaces)->copyFrom( faces, false );
        (*ppFaceClusters)->copyFrom( fc, false );
        
        // get rid of temp data
        delete [] nodes;
        delete [] faces;
        delete [] fc;

        // statistics
    #if RL_PRINT_STATS
        float nodeSize = (float)sizeof(BvhNode)*nodeIndexer/1024/1024;
        float facesize = (float)sizeof(Face)*faceIndexer/1024/1024;
        float faceClusterSize = (float)sizeof(FaceCluster)*faceClusterIndexer/1024/1024;
        printf("BVH NodeCount %d, size %fmb\n", nodeIndexer, nodeSize );
        printf("BVH FaceCount %d, size %fmb\n", faceIndexer, facesize );
        printf("BVH FaceClusterCount %d, size %fmb\n", faceClusterIndexer, faceClusterSize );
        printf("BVH Total: %f\n", (nodeSize+facesize+faceClusterSize));
    #endif

        return ERROR_ALL_FINE;
    }

}