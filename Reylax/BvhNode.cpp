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

        const u32 stSize = 64;
        stNode stack[stSize];

        // push first faces on stack and approximate face count to allocate
        u32 allocatedFaceCount = 0;
        forEachFace(meshData, numMeshDatas, [&](u32 mId, const u32 id[3], const vec3 v[3])
        {
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

        // allocate sufficient amount of data
        u32 numNodes = 1024*1024*8;
        u32 numFacesClusters = _max<u32>((u32)sqrt(numNodes), allocatedFaceCount/(BVH_NUM_FACES_IN_LEAF/8));

    #if RL_PRINT_STATS
        printf("--- BVH Static scene allocations ---\n\n");
        float bvhSize   = (float)sizeof(BvhNode)*numNodes/1024/1024;
        float faceSize  = (float)sizeof(Face)*allocatedFaceCount/1024/1024;
        float fclusSize = (float)sizeof(BvhNode)*numFacesClusters/1024/1024;
        printf("Intermediate data\n");
        printf("BvhNodes: %d, size %.3fmb\n", numNodes, bvhSize);
        printf("Faces: %d, size %.3fmb\n", allocatedFaceCount, faceSize);
        printf("Fclusters: %d, size %.3fmb\n", numFacesClusters, fclusSize);
        printf("Total: %.3fmb\n", (bvhSize+faceSize+fclusSize));
    #endif

        BvhNode* nodes  = new BvhNode[numNodes];
        Face* faces     = new Face[allocatedFaceCount]; // times 2 to accomodate for overlapping faces on two boxes
        FaceCluster* fc = new FaceCluster[numFacesClusters];

        // indexers into memory buffers
        u32 nodeIndexer = 0;
        u32 faceIndexer = 0;
        u32 faceClusterIndexer = 0;
        
        // setup remaining part of first node in stack (faces for stack[0] have been poplated already)
        stNode* st = stack;
        st->parentIdx = RL_INVALID_INDEX;
        st->depth = 0;
        i32 top=0;

        // set bbox of first stack node
        vec3 centre;
        determineBbox(st, meshData, st->bMin, st->bMax, centre);

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

            // determine new bbox of popped node and refit to smallest possible
            vec3 bMin, bMax;
            determineBbox(st, meshData, bMin, bMax, centre);
            bMin = max( bMin, st->bMin );
            bMax = min( bMax, st->bMax );

            // bbox of new node is now determined
            BvhNode* node = nodes + nodeIndexer++;
            node->cp = centre;
            node->hs = (bMax-bMin)*0.5001f;
            node->left  = RL_INVALID_INDEX;
            node->right = RL_INVALID_INDEX;

            if ( st->depth == BVH_MAX_DEPTH || (u32)st->faces.size() <= BVH_NUM_FACES_IN_LEAF )
            {
                if ( (u32)st->faces.size() > BVH_NUM_FACES_IN_LEAF )
                {
                    printf("Could not fit all faces (%d) in a leaf node of BVH while max depth was reached.\n", (u32)st->faces.size());
                }
                else
                {
                #if BVH_DBG_INFO
                    printf("Fitted %d faces in leaf, depth %d\n", (u32)st->faces.size(), st->depth);
                #endif
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
                u32 splitAxis = 0;
                float biggest = node->hs.x;
                if ( node->hs.y > biggest ) { splitAxis = 1; biggest = node->hs.y; }
                if ( node->hs.z > biggest ) { splitAxis = 2; }
            #if BVH_DBG_INFO
               printf("splitAxis %d\n", splitAxis);
            #endif

                vec3 dbox = bMax-bMin;
                assert( dbox.x>0.f && dbox.y>0.f && dbox.z>0.f );
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

            #if BVH_DBG_INFO
                printf("Faces left: %zd | cpL %.3f %.3f %.3f | hsL %.3f %.3f %.3f\n", facesL.size(), cpL.x, cpL.y, cpL.z, hsL.x, hsL.y, hsL.z);
                printf("Faces right: %zd | cpR %.3f %.3f %3f | hsR %.3f %.3f %.3f\n", facesR.size(), cpR.x, cpR.y, cpR.z, hsR.x, hsR.y, hsR.z);
            #endif

                u32 depth = st->depth;
                assert(top+2 < stSize);

                // right
                st = &stack[++top];
                st->depth = depth+1;
                st->parentIdx = nodeIndexer-1;
                st->faces = move(facesR);
                st->bMin  = rMin;
                st->bMax  = bMax;
                // left
                st = &stack[++top];
                st->depth = depth+1;
                st->parentIdx = nodeIndexer-1;
                st->faces = move(facesL);
                st->bMin  = bMin;
                st->bMax  = lMax;
            } // split
        } // while

        // setup device memory
        *ppBvhTree = new DeviceBuffer( sizeof(BvhNode) * nodeIndexer );
        *ppFaces   = new DeviceBuffer( sizeof(Face) * faceIndexer );
        *ppFaceClusters = new DeviceBuffer( sizeof(FaceCluster) * faceClusterIndexer );

        (*ppBvhTree)->copyFrom( nodes, false );
        (*ppFaces)->copyFrom( faces, false );
        (*ppFaceClusters)->copyFrom( fc, false );

        // statistics
    #if RL_PRINT_STATS
        float nodeSize = (float)sizeof(BvhNode)*nodeIndexer/1024/1024;
        float facesize = (float)sizeof(Face)*faceIndexer/1024/1024;
        float faceClusterSize = (float)sizeof(FaceCluster)*faceClusterIndexer/1024/1024;
        printf("\nActual allocations on device\n");
        printf("NodeCount %d, size %.3fmb\n", nodeIndexer, nodeSize);
        printf("FaceCount %d, size %.3fmb\n", faceIndexer, facesize);
        printf("FaceClusterCount %d, size %.3fmb\n", faceClusterIndexer, faceClusterSize);
        printf("Total: %.3fmb\n", (nodeSize+facesize+faceClusterSize));
        showDebugInfo( nodes );
    #endif

        // Ensure data is copied to device before deletion of host memory
        syncDevice();

        // get rid of intermediate data
        delete[] nodes;
        delete[] faces;
        delete[] fc;

    #if RL_PRINT_STATS
        printf("\nDELETED intermediate allocations\n\n");
        printf("--- End BVH static scene creation ---\n\n");
    #endif

        return ERROR_ALL_FINE;
    }

    void BvhNode::determineBbox(stNode* st, const MeshData** meshData, vec3& bMin, vec3& bMax, vec3& centre)
    {
        bMin = vec3(FLT_MAX);
        bMax = vec3(FLT_MIN);
        centre = vec3(0);
        for ( auto& f : st->faces )
        {
            const vec3* vd  = (const vec3*)(meshData[f.w])->vertexData[VERTEX_DATA_POSITION];
            const vec3 v[3] ={ vd[f.x], vd[f.y], vd[f.z] };
            for ( auto& vi: v )
            {
                centre += vi;
                bMin = _min<vec3>(bMin, vi);
                bMax = _max<vec3>(bMax, vi);
            }
        }
        centre /= (float)st->faces.size()*3;
    }

    void BvhNode::showDebugInfo(const BvhNode* nodes)
    {
        struct stackNode
        {
            u32 node;
            u32 depth;
        };

        stackNode stack[64];
        i32 top=0;
        stack[0].depth = 0;
        stack[0].node  = 0;

        u64 avgDepth = 0;
        u64 numLeafs = 0;
        u32 maxDepth = 0;
        u64 numFaces = 0;
        u32 numNodes = 0;
        u64 avgFacesPerLeaf = 0;
        u64 maxFacesInLeaf  = 0;

        while ( top >= 0 )
        {
            stackNode* st = &stack[top--];
            const BvhNode* node = nodes + st->node;
            maxDepth = max(maxDepth, st->depth);
            numNodes++;

            if ( node->isLeaf() )
            {
                numLeafs++;
                numFaces += node->numFaces();
                avgFacesPerLeaf += node->numFaces();
                maxFacesInLeaf  = max<u64>(node->numFaces(), maxFacesInLeaf);
                avgDepth += st->depth;
            }
            else
            {
                u32 depth = st->depth;
                st = &stack[++top];
                st->node  = node->left;
                st->depth = depth+1;
                st = &stack[++top];
                st->node  = node->right;
                st->depth = depth+1;
            }
        }

        printf("NumNodes %d, MaxDepth %d, AvgDepth %zd, NumLeafs %zd, NumFaces %zd, avgFacesPerLeaf %zd, maxFacesInLeaf %zd\n",
               numNodes, maxDepth, avgDepth/numLeafs, numLeafs, numFaces, avgFacesPerLeaf/numLeafs, maxFacesInLeaf);
    }
    
}