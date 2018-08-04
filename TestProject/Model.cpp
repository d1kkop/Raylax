#include "Reylax.h"
#include "glm/common.hpp"
#include "glm/geometric.hpp"
#include "glm/vec3.hpp"
#include "glm/mat3x3.hpp"
#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"
#include <iostream>
#include <cassert>
#include <vector>
using namespace Assimp;
using namespace std;
using namespace Reylax;
using namespace glm;


float _min2(float a, float b) { return a < b ? a : b; }
float _max2(float a, float b) { return a > b ? a : b; }
vec3 _min2(const vec3& a, const vec3& b) { return vec3(_min2(a.x, b.x), _min2(a.y, b.y), _min2(a.z, b.z)); }
vec3 _max2(const vec3& a, const vec3& b) { return vec3(_max2(a.x, b.x), _max2(a.y, b.y), _max2(a.z, b.z)); }

bool loadModel(const std::string& name, vector<IMesh*>& meshes)
{
    // Create an instance of the Importer class
    Importer importer;
    // And have it read the given file with some example postprocessing
    // Usually - if speed is not the most important aspect for you - you'll
    // probably to request more postprocessing than we do in this example.
    const aiScene* scene = importer.ReadFile(name,
                                                aiProcess_CalcTangentSpace       |
                                                aiProcess_Triangulate            |
                                                aiProcess_JoinIdenticalVertices  |
                                                aiProcess_SortByPType);
    // If the import failed, report it
    if ( !scene ) {
        cout << importer.GetErrorString() << endl;
        assert(false);
        return false;
    }
    // Now we can access the file's contents.
    u32 totalVertices =0;
    u32 totalFaces =0;
    vec3 bMin(FLT_MAX);
    vec3 bMax(FLT_MIN);
    for (u32 i = 0; i < scene->mNumMeshes ; i++)
    {
        const aiMesh* aiMesh = scene->mMeshes[i];
        auto* rlMesh = Reylax::IMesh::create();
        meshes.push_back( rlMesh );
        totalVertices += aiMesh->mNumVertices;
        totalFaces += aiMesh->mNumFaces;
        u32 err=0;
        // deform indices
        u32* indices = new u32[aiMesh->mNumFaces*3];
        for ( u32 i=0; i<aiMesh->mNumFaces; ++i )
        {
            assert(aiMesh->mFaces[i].mNumIndices==3);
            indices[i*3+0] = aiMesh->mFaces[i].mIndices[0];
            indices[i*3+1] = aiMesh->mFaces[i].mIndices[1];
            indices[i*3+2] = aiMesh->mFaces[i].mIndices[2];
            assert(indices[i*3+0]  < aiMesh->mNumVertices);
            assert(indices[i*3+1]  < aiMesh->mNumVertices);
            assert(indices[i*3+2]  < aiMesh->mNumVertices);
            vec3 v[3] =
            {
                *(vec3*)&aiMesh->mVertices[indices[i*3+0]],
                *(vec3*)&aiMesh->mVertices[indices[i*3+1]],
                *(vec3*)&aiMesh->mVertices[indices[i*3+2]]
            };
            for ( auto& vs : v )
            {
                bMin = _min2(bMin, vs);
                bMax = _max2(bMax, vs);
            }
        }
        rlMesh->setIndices( indices, aiMesh->mNumFaces*3 );
        delete [] indices;
        assert(err==0);
        err = rlMesh->setVertexData( (float*)(aiMesh->mVertices), aiMesh->mNumVertices, 3, VERTEX_DATA_POSITION );
        assert(err==0);
        if ( aiMesh->HasNormals() )
        {
            err = rlMesh->setVertexData( (float*)(aiMesh->mNormals), aiMesh->mNumVertices, 3, VERTEX_DATA_NORMAL );
            assert(err==0);
        }
        float* uv = new float[aiMesh->mNumVertices*2];
        for ( u32 k=0; k<2; k++ )
        {
            if ( aiMesh->HasTextureCoords(k) )
            {
                // deform to 2 based component
                for ( u32 j = 0; j < aiMesh->mNumVertices; j++ )
                {
                    uv[j*2]   = aiMesh->mTextureCoords[k][j].x;
                    uv[j*2+1] = aiMesh->mTextureCoords[k][j].y;
                }
                err = rlMesh->setVertexData(uv, aiMesh->mNumVertices, 2, VERTEX_DATA_UV1+k);
                assert(err==0);
            }
        }
        delete[] uv;
        if ( aiMesh->HasTangentsAndBitangents() )
        {
            err = rlMesh->setVertexData((float*)(aiMesh->mTangents), aiMesh->mNumVertices, 3, VERTEX_DATA_TANGENT);
            assert(err==0);
            err = rlMesh->setVertexData((float*)(aiMesh->mBitangents), aiMesh->mNumVertices, 3, VERTEX_DATA_BITANGENT);
            assert(err==0);
        }
    }
    printf("\n\n");
    printf("Model: %s\n", name.c_str());
    printf("Faces: %fM\n", (float)totalFaces/(1000000));
    printf("Vertices: %fM\n", (float)totalVertices/(1000000));
    printf("bMin %.3f %.3f %.3f\n", bMin.x, bMin.y, bMin.z);
    printf("bMax %.3f %.3f %.3f\n", bMax.x, bMax.y, bMax.z);
    vec3 dt = bMax-bMin;
    printf("Size %.3f %.3f %.3f\n", dt.x, dt.y, dt.z);
    printf("\n");
    // We're done. Everything will be cleaned up by the importer destructor
    return true;
}
