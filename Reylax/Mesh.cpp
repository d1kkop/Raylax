#include "Mesh.h"
#include <cassert>
#include <memory>
using namespace std;


namespace Reylax
{
    // ------ IMesh ----------------------------------------------------------------------------------------

    IMesh* IMesh::create()
    {
        return new Mesh();
    }

    // ------ Mesh ----------------------------------------------------------------------------------------

    Mesh::Mesh()
    {
        d.numVertices = 0;
        d.numIndices = 0;
        d.indices = nullptr;
        for (auto& p : d.vertexData) p = nullptr;
        for (u32 & vds : d.vertexDataSizes) vds = 0;
    }

    Mesh::~Mesh()
    {
        delete [] d.indices;
        for ( auto& p : d.vertexData ) delete [] p;
    }

    u32 Mesh::setVertexData(const float* vertexData, u32 numVertices, u32 numComponents, u32 slotId)
    {
        if ( !vertexData || numVertices==0 || slotId >= VERTEX_DATA_COUNT || numComponents > 4 ||
             (d.numVertices!=0 && d.numVertices!=numVertices) ||
             (slotId == VERTEX_DATA_POSITION && numComponents != 3 )
            )
        {
            return ERROR_INVALID_PARAMETER;
        }
        d.vertexData[slotId] = new float[numComponents*numVertices];
        if ( !d.vertexData[slotId] ) return ERROR_GPU_ALLOC_FAIL;
        memcpy( d.vertexData[slotId], vertexData, numComponents*numVertices*sizeof(float) );
        d.vertexDataSizes[slotId] = numComponents;
        return ERROR_ALL_FINE;
    }

    u32 Mesh::setIndices(const u32* indices, u32 numIndices)
    {
        if ( !indices || (numIndices%3)!=0 ) return ERROR_INVALID_PARAMETER;
        d.numIndices = numIndices;
        d.indices = new u32[ d.numIndices ];
        if ( !d.indices ) return ERROR_GPU_ALLOC_FAIL;
        memcpy( d.indices, indices, numIndices*sizeof(u32) );
        return ERROR_ALL_FINE;
    }
}
