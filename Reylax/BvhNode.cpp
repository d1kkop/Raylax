#include "ReylaxCommon.h"
#include "Reylax_internal.h"
#include "Reylax.h"
#include "DeviceBuffer.h"
using namespace std;


int rlTriBoxOverlap(const float* boxcenter, const float* boxhalfsize, const vec3 triverts[3]);


namespace Reylax
{
    // -------- BvhNode -------------------------------------------------------------------------------------------------

    u32 BvhNode::build(const MeshData** meshPtrs, u32 numMeshDatas,
                       DeviceBuffer** ppBvhTree,
                       DeviceBuffer** ppFaces,
                       DeviceBuffer** ppFaceClusters,
                       DeviceBuffer** ppSides,
                       vec3& worldMin,
                       vec3& worldMax,
                       SplitFunction splitFunc,
                       SidesFunction sidesFunc)
    {
        if ( !meshPtrs || numMeshDatas == 0 || !ppBvhTree || !ppFaces || !ppFaceClusters || !ppSides || !splitFunc || !sidesFunc )
        {
            return ERROR_INVALID_PARAMETER;
        }

        struct stNode
        {
            u32 parentIdx, depth, spAxis;
            std::vector<Face> faces;
            vec3 bMin, bMax;
        };

        const u32 stSize = BVH_MAX_DEPTH;
        stNode stack[stSize];

        // push first faces on stack and approximate face count to allocate
        u32 allocatedFaceCount = 0;
        forEachFace(meshPtrs, numMeshDatas, [&](u32 mId, const u32 id[3], const vec3 v[3])
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
        allocatedFaceCount *= 30;

        // allocate sufficient amount of data
        u32 numNodes = 1024*1024*8;
        u32 numFacesClusters = numNodes*2/BVH_NUM_FACES_IN_LEAF;

    #if RL_PRINT_STATS
        printf("--- BVH Static intermediate allocations ---\n\n");
        float bvhSize   = (float)sizeof(BvhNode)*numNodes/1024/1024;
        float faceSize  = (float)sizeof(Face)*allocatedFaceCount/1024/1024;
        float fclusSize = (float)sizeof(BvhNode)*numFacesClusters/1024/1024;
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
        st->depth  = 0;
        st->spAxis = RL_INVALID_INDEX;
        i32 top=0;

        // set bbox of first stack node
        determineBbox(st->faces, meshPtrs, st->bMin, st->bMax);
        st->bMin-=vec3(0.001f); // TODO <--- hmmm
        st->bMax+=vec3(0.001f);
        worldMin = st->bMin;
        worldMax = st->bMax;

        // generate tree
        while ( top >= 0 )
        {
            st = &stack[top--];

            // bbox of new node is now determined
            assert(nodeIndexer+1 < numNodes);
            BvhNode*node = nodes + nodeIndexer++;
            node->bMin   = st->bMin;
            node->bMax   = st->bMax;
            node->left   = 0;   // 0 is also useful to detect as invalid because only the first node can have 0, all others must be higher
            node->right  = 0;

            vec3 hs = (st->bMax - st->bMin)*.5f;
            assert(hs.x>0.f && hs.y>0.f && hs.z>0.f);

            // If cur stack node has valid parent index, we can now efficiently assign index children of parent
            // such that the indexing of nodes: 0, 1, 2... is linear in memory.
            if ( RL_VALID_INDEX(st->parentIdx) )
            {
                BvhNode* parent = nodes + st->parentIdx;
                if ( parent->left==0 ) parent->left = nodeIndexer-1;
                else
                {
                    assert( BVH_GET_INDEX(parent->right)==0 );
                    BVH_SET_INDEX(parent->right, nodeIndexer-1);
                }
            }

            if ( st->depth == BVH_MAX_DEPTH-1 || (u32)st->faces.size() <= BVH_NUM_FACES_IN_LEAF || 
                ( hs.x <= BVH_MIN_SIZE || hs.y <= BVH_MIN_SIZE || hs.z <= BVH_MIN_SIZE ) )
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
                u32 numFaces = _min( (u32)st->faces.size(), (u32)BVH_NUM_FACES_IN_LEAF );
                BVH_SET_LEAF_AND_FACES( node->left, numFaces );
                node->right = faceClusterIndexer++;
                FaceCluster* cluster = fc + node->right;
                cluster->numFaces    = numFaces;
                assert( node->numFaces()==cluster->numFaces );
                assert( BVH_GETNUM_TRIANGLES(node->left)==cluster->numFaces );
                for ( u32 i=0; i<cluster->numFaces; ++i )
                {
                    assert( faceIndexer < allocatedFaceCount );
                    Face* fNew = faces + faceIndexer++;
                    *fNew = st->faces[i];
                    cluster->faces[i] = faceIndexer-1;
                }
            }
            else
            {
                u32 splitAxis = 0;
                float s = 0;
                splitFunc( meshPtrs, st->faces, st->bMin, st->bMax, s, splitAxis );
                BVH_SET_AXIS( node->right, splitAxis );
            #if BVH_HAS_SPLIT
                node->split = s;
            #endif

            #if BVH_DBG_INFO
               printf("splitAxis %d\n", splitAxis);
            #endif

                vec3 lMax = st->bMax;
                vec3 rMin = st->bMin;
                lMax[splitAxis] = s;
                rMin[splitAxis] = s;

                vec3 cpL = (st->bMin+lMax)*.5f;
                vec3 hsL = (lMax-st->bMin)*.5f;
                vec3 cpR = (rMin+st->bMax)*.5f;
                vec3 hsR = (st->bMax-rMin)*.5f;

                assert(hsL.x>0.f && hsL.y>0.f && hsL.z>0.f);
                assert(hsR.x>0.f && hsR.y>0.f && hsR.z>0.f);

                vector<Face> facesL, facesR;
                for ( auto& f : st->faces )
                {
                    const vec3* vd  = ( const vec3*)(meshPtrs[f.w] )->vertexData[VERTEX_DATA_POSITION];
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
                vec3 pBmax = st->bMax;
                vec3 pBmin = st->bMin;

                // right
                st = &stack[++top];
                st->depth = depth+1;
                st->parentIdx = nodeIndexer-1;
                st->faces = move(facesR);
                st->bMin  = rMin;
                st->bMax  = pBmax;
                st->spAxis = splitAxis;

                // left
                st = &stack[++top];
                st->depth = depth+1;
                st->parentIdx = nodeIndexer-1;
                st->faces = move(facesL);
                st->bMin  = pBmin;
                st->bMax  = lMax;
                st->spAxis = splitAxis;

            } // split
        } // while

        // Sides of leafs can now be determine as index of every node is known.
        u32* sides_data = new u32[faceClusterIndexer*6];
        sidesFunc( sides_data, nodes, faceClusterIndexer );

        // setup device memory
        *ppBvhTree = new DeviceBuffer(sizeof(BvhNode) * nodeIndexer);
        *ppFaces   = new DeviceBuffer(sizeof(Face) * faceIndexer);
        *ppFaceClusters = new DeviceBuffer(sizeof(FaceCluster) * faceClusterIndexer);
        *ppSides   = new DeviceBuffer(sizeof(u32)*6*faceClusterIndexer);

        (*ppBvhTree)->copyFrom(nodes, false);
        (*ppFaces)->copyFrom(faces, false);
        (*ppFaceClusters)->copyFrom(fc, false);
        (*ppSides)->copyFrom( sides_data, false );

        // statistics
    #if RL_PRINT_STATS
        float nodeSize = (float)sizeof(BvhNode)*nodeIndexer/1024/1024;
        float facesize = (float)sizeof(Face)*faceIndexer/1024/1024;
        float faceClusterSize = (float)sizeof(FaceCluster)*faceClusterIndexer/1024/1024;
        float sidesSize = (float)sizeof(u32)*6*faceClusterIndexer/1024/1024;
        printf("\nActual allocations on device\n");
        printf("NodeCount %d, size %.3fmb\n", nodeIndexer, nodeSize);
        printf("FaceCount %d, size %.3fmb\n", faceIndexer, facesize);
        printf("FaceClusterCount %d, size %.3fmb\n", faceClusterIndexer, faceClusterSize);
        printf("Sidelinks %d, size %.3fmb\n", faceClusterIndexer, sidesSize);
        printf("Total: %.3fmb\n", (nodeSize+facesize+faceClusterSize+sidesSize));
        showDebugInfo( nodes );
    #endif

        // Ensure data is copied to device before deletion of host memory
        SyncDevice();

        // get rid of intermediate data
        delete[] nodes;
        delete[] faces;
        delete[] fc;
        delete[] sides_data;

    #if RL_PRINT_STATS
        printf("\nDELETED intermediate allocations\n\n");
        printf("--- End BVH static scene creation ---\n\n");
    #endif

        return ERROR_ALL_FINE;
    }

    void BvhNode::determineCentre(vector<Face>& faces, const MeshData** meshData, vec3& centre)
    {
        centre = vec3(0);
        for ( auto& f : faces )
        {
            const vec3* vd  = (const vec3*)(meshData[f.w])->vertexData[VERTEX_DATA_POSITION];
            const vec3 v[3] = { vd[f.x], vd[f.y], vd[f.z] };
            for ( auto& vi: v )
            {
                centre += vi;
            }
        }
        centre /= faces.size()*3;
    }

    void BvhNode::determineBbox(vector<Face>& faces, const MeshData** meshPtrs, vec3& bMin, vec3& bMax)
    {
        bMin = vec3(FLT_MAX);
        bMax = vec3(FLT_MIN);
        for ( auto& f : faces )
        {
            const vec3* vd  = (const vec3*)(meshPtrs[f.w])->vertexData[VERTEX_DATA_POSITION];
            const vec3 v[3] = { vd[f.x], vd[f.y], vd[f.z] };
            for ( auto& vi: v )
            {
                bMin = _min<vec3>(bMin, vi);
                bMax = _max<vec3>(bMax, vi);
            }
        }
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
            maxDepth = _max(maxDepth, st->depth);
            numNodes++;

            if ( node->isLeaf() )
            {
                numLeafs++;
                numFaces += node->numFaces();
                avgFacesPerLeaf += node->numFaces();
                maxFacesInLeaf  = _max<u64>(node->numFaces(), maxFacesInLeaf);
                avgDepth += st->depth;
            }
            else
            {
                u32 depth = st->depth;
                st = &stack[++top];
                st->node  = BVH_GET_INDEX( node->left );
                st->depth = depth+1;
                st = &stack[++top];
                st->node  = BVH_GET_INDEX( node->right );
                st->depth = depth+1;
            }
        }

        printf("NumNodes %d, MaxDepth %d, AvgDepth %zd, NumLeafs %zd, NumFaces %zd, avgFacesPerLeaf %zd, maxFacesInLeaf %zd\n",
               numNodes, maxDepth, avgDepth/numLeafs, numLeafs, numFaces, avgFacesPerLeaf/numLeafs, maxFacesInLeaf);
    }
    
}