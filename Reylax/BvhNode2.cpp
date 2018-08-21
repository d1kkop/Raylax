#include "ReylaxCommon.h"
#include "Reylax_internal.h"
#include "Reylax.h"
#include "DeviceBuffer.h"
using namespace std;


int rlTriBoxOverlap(const float* boxcenter, const float* boxhalfsize, const vec3 triverts[3]);


namespace Reylax
{
    // -------- BvhNode -------------------------------------------------------------------------------------------------

    u32 BvhNode::build2(const MeshData** meshPtrs, u32 numMeshDatas,
                       DeviceBuffer** ppBvhTree,
                       DeviceBuffer** ppFaces,
                       DeviceBuffer** ppFaceClusters,
                       DeviceBuffer** ppSides,
                       vec3& worldMin,
                       vec3& worldMax)
    {
        if ( !meshPtrs || numMeshDatas == 0 || !ppBvhTree || !ppFaces || !ppFaceClusters || !ppSides )
        {
            return ERROR_INVALID_PARAMETER;
        }

        struct stNode
        {
            u32 depth;
            BvhNode* node;
            std::vector<Face> faces;
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
        allocatedFaceCount *= 200;

        // allocate sufficient amount of data
        u32 numNodes = 1024*1024*8;
        u32 numFacesClusters = _max<u32>((u32)sqrt(numNodes), allocatedFaceCount/(BVH_NUM_FACES_IN_LEAF/8));

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
        u32 nodeIndexer = 1;
        u32 faceIndexer = 0;
        u32 faceClusterIndexer = 0;
        
        // setup remaining part of first node in stack (faces for stack[0] have been poplated already)
        stNode* st = stack;
        st->depth  = 0;
        st->node   = nodes;
        i32 top=0;

        // set bbox of first stack node
        determineBbox(st->faces, meshPtrs, worldMin, worldMax);
        st->node->bMin = worldMin;
        st->node->bMax = worldMax;
        st->node->left  = 0;
        st->node->right = 0;

        // generate tree
        while ( top >= 0 )
        {
            st = &stack[top--];
            BvhNode*node = st->node;

            vec3 bMin = node->bMin;
            vec3 bMax = node->bMax;
            vec3 hs = (bMax - bMin)*.5f;
            assert(hs.x>0.f && hs.y>0.f && hs.z>0.f);

            if ( st->depth == BVH_MAX_DEPTH || (u32)st->faces.size() <= BVH_NUM_FACES_IN_LEAF ||
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
                float biggest = hs.x;
                if ( hs.y > biggest ) { splitAxis = 1; biggest = hs.y; }
                if ( hs.z > biggest ) { splitAxis = 2; }

            #if BVH_DBG_INFO
               printf("splitAxis %d\n", splitAxis);
            #endif

                float s = (bMax[splitAxis] + bMin[splitAxis])*.5f;
                vec3 lMax = node->bMax;
                vec3 rMin = node->bMin;
                lMax[splitAxis] = s;
                rMin[splitAxis] = s;

                vec3 cpL = (bMin+lMax)*.5f;
                vec3 hsL = (lMax-bMin)*.5f;
                vec3 cpR = (rMin+bMax)*.5f;
                vec3 hsR = (bMax-rMin)*.5f;

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

                // left
                BvhNode* left = nodes + nodeIndexer++;
                left->bMin  = bMin;
                left->bMax  = lMax;
                left->left  = 0;
                left->right = 0;

                // right
                BvhNode* right = nodes + nodeIndexer++;
                right->bMin  = rMin;
                right->bMax  = bMax;
                right->left  = 0;
                right->right = 0;

                // push new nodes on stack
                u32 depth = st->depth;
                assert(top+2 < stSize);

                st = &stack[++top];
                st->depth = depth+1;
                st->faces = move(facesR);
                st->node  = right;

                st = &stack[++top];
                st->depth = depth+1;
                st->faces = move(facesL);
                st->node  = left;

                // left and right of parent are now known
                BVH_SET_AXIS( node->right, splitAxis );
                BVH_SET_INDEX( node->right, nodeIndexer-1 );
                node->left = nodeIndexer-2;

            } // split
        } // while

        // Sides of leafs can now be determine as index of every node is known.
        u32* sides_data = new u32[faceClusterIndexer*6];
        struct sideStack
        {
            u32 node;
            u32 indices[6];
        };
        sideStack sstack[stSize];
        sideStack* sst = &sstack[0];
        sst->node = 0;
        top=0;
        for ( auto& sd : sst->indices ) sd = RL_INVALID_INDEX;
        while ( top>=0 )
        {
            sst = &sstack[top--];
            BvhNode* node = nodes + sst->node;
            if ( node->isLeaf() )
            {
                assert( node->right < faceClusterIndexer );
                u32* pSides = sides_data + node->right*6;
                memcpy( pSides, sst->indices, sizeof(u32)*6 );
            }
            else
            {
                assert( top + 2 < stSize );
                u32 spAxis = BVH_GET_AXIS( node->right );
                assert(spAxis==0 || spAxis==1 || spAxis==2);
                u32 oldSides[6];
                memcpy( oldSides, sst->indices, sizeof(u32)*6 );
                
                // right
                sst = &sstack[++top];
                sst->node = BVH_GET_INDEX(node->right);
                memcpy( sst->indices, oldSides, sizeof(u32)*6 );
                sst->indices[ spAxis*2 ] = node->left;
                //printf("Indices Right\n");
                //for ( auto& i : sst->indices ) printf("%d ", i );
                //printf("\n");
                // left
                sst = &sstack[++top];
                sst->node = node->left;
                memcpy( sst->indices, oldSides, sizeof(u32)*6 );
                sst->indices[ spAxis*2+1 ] = BVH_GET_INDEX( node->right ); // spAxis also stored in right
                //printf("Indices Left\n");
                //for ( auto& i : sst->indices ) printf("%d ", i );
                //printf("\n");
            }
        }

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
}