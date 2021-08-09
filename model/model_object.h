#pragma once
#include "model_data.h"
#include "shading/vertex_program.h"
#include "shading/pixel_program.h"

struct ModelObject
{
    ModelData modelData;
    std::unique_ptr<VertexProgram> vertexProgram;
    std::unique_ptr<PixelProgram> pixelProgram;
};
