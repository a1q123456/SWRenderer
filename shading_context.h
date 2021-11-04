#pragma onec
#include "model/vertex.h"
#include "shading/pixel_program.h"
#include "shading/vertex_program.h"


struct ProgramContext
{
    int vsOutputPosIdx = 0;
    int vsOutputUvIdx = 0;
    int vsOutputColorIdx = 0;
    VertexAttributeTypes vsOutputPosType;
    VertexAttributeTypes vsOutputUvType;
    VertexAttributeTypes vsOutputColorType;
    std::vector<VertexDataDescriptor> vsInputDesc;
    std::vector<VertexDataDescriptor> vsOutputDesc;
    std::vector<VertexDataDescriptor> psInputDesc;

    uint32_t inputVertexAttributes;
    VertexFunction vertexEntry;
    PixelFunction pixelEntry;

    VertexProgram *vertexProgram = nullptr;
    PixelProgram *pixelProgram = nullptr;

    bool vsOutputsUv = false;
    bool vsOutputsColor = false;
    std::map<int, int> psVsIndexMap;
};
