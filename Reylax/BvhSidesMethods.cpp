#include "ReylaxCommon.h"
#include "Reylax_internal.h"
#include "Reylax.h"
#include "DeviceBuffer.h"
using namespace std;


#define BVH_SIDES_SCALE 0.9999f
#define BVH_SIDES_PUSH 0.0001f


namespace Reylax
{
    void BvhNode::sidesDefault(u32* sides_data, const BvhNode* nodes, u32 numLeafs)
    {
        struct sideStack
        {
            u32 node;
            u32 indices[6];
        };
        sideStack sstack[BVH_MAX_DEPTH];
        sideStack* sst = &sstack[0];
        sst->node = 0;
        i32 top=0;
        for ( auto& sd : sst->indices ) sd = RL_INVALID_INDEX;
        while ( top>=0 )
        {
            sst = &sstack[top--];
            const BvhNode* node = nodes + sst->node;
            if ( node->isLeaf() )
            {
                assert(node->right < numLeafs);
                u32* pSides = sides_data + node->right*6;
                memcpy(pSides, sst->indices, sizeof(u32)*6);
            }
            else
            {
                assert(top + 2 < BVH_MAX_DEPTH);
                u32 spAxis = BVH_GET_AXIS(node->right);
                assert(spAxis==0 || spAxis==1 || spAxis==2);
                u32 oldSides[6];
                memcpy(oldSides, sst->indices, sizeof(u32)*6);

                // right
                sst = &sstack[++top];
                sst->node = BVH_GET_INDEX(node->right);
                memcpy(sst->indices, oldSides, sizeof(u32)*6);
                sst->indices[spAxis*2] = node->left;
                //printf("Indices Right\n");
                //for ( auto& i : sst->indices ) printf("%d ", i );
                //printf("\n");
                // left
                sst = &sstack[++top];
                sst->node = node->left;
                memcpy(sst->indices, oldSides, sizeof(u32)*6);
                sst->indices[spAxis*2+1] = BVH_GET_INDEX(node->right); // spAxis also stored in right
                //printf("Indices Left\n");
                //for ( auto& i : sst->indices ) printf("%d ", i );
                //printf("\n");
            }
        }
    }

    u32 findSideInBbox(const vec3 sideMinMax[2], u32 self, const BvhNode* nodes)
    {
        u32 side = RL_INVALID_INDEX;
        u32 stack[BVH_MAX_DEPTH];
        stack[0] = 0;
        i32 top=0;
        while ( top>= 0 )
        {
            u32 stNode = stack[top--];
            if ( stNode == self ) continue;
            const BvhNode* node = nodes + stNode;
            vec3 bmin = node->bMin;
            vec3 bmax = node->bMax;
            if ( AABBInAABB( sideMinMax[0], sideMinMax[1], bmin, bmax ) )
            // if ( PointInAABB(sideMinMax[0], bmin, bmax) && PointInAABB(sideMinMax[1], bmin, bmax) )
            {
                side = stNode;
                if ( !node->isLeaf() )
                {
                    assert(top+2 < BVH_MAX_DEPTH);
                    stack[++top] = node->left;
                    stack[++top] = BVH_GET_INDEX( node->right );
                }
            }
        }
        return side;
    }

    void BvhNode::sidesBbox(u32*sides_data, const BvhNode* nodes, u32 numLeafs)
    {
        u32 stack[BVH_MAX_DEPTH];
        i32 top=0;
        stack[0] = 0;
        while ( top>=0 )
        {
            u32 stNode = stack[top--];
            const BvhNode* node = nodes + stNode;
            if ( node->isLeaf() )
            {
                assert(node->right < numLeafs);
                u32* pSides = sides_data + node->right*6;
                vec3 bmin = node->bMin;
                vec3 bmax = node->bMax;
                
                for ( u32 j=0; j<3; ++j ) // axis
                {
                    for ( u32 i=0; i<2; ++i ) // min/max
                    {
                        // For each side in leaf
                        vec3 tbbox []={ bmin, bmax };
                        vec3 cp = (tbbox[0]+tbbox[1])*.5f;
                        vec3 hs = (tbbox[1]-tbbox[0])*.5f;
                        tbbox[0] -= cp; // to origin
                        tbbox[1] -= cp;
                        tbbox[0][(j+1)%3] *= BVH_SIDES_SCALE;
                        tbbox[1][(j+1)%3] *= BVH_SIDES_SCALE;
                        tbbox[0][(j+2)%3] *= BVH_SIDES_SCALE;
                        tbbox[1][(j+2)%3] *= BVH_SIDES_SCALE;
                        tbbox[(i+1)&1][j]  = tbbox[i][j];
                        tbbox[0][j] += hs[j] * BVH_SIDES_PUSH * (i==0?-1.f:1.f);
                        tbbox[1][j] += hs[j] * BVH_SIDES_PUSH * (i==0?-1.f:1.f);
                        tbbox[0] += cp; // place back in world
                        tbbox[1] += cp;
                        pSides[  j*2 + i ] = findSideInBbox( tbbox, stNode, nodes );
                    }
                }
            }
            else
            {
                assert(top + 2 < BVH_MAX_DEPTH);
                stack[++top] = node->left;
                stack[++top] = BVH_GET_INDEX(node->right);
            }
        }
    }
}