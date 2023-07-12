#pragma once
#include "model/vertex.h"
#include "shading/light/light.h"
#include "model/data_pack.h"

class PixelProgram;
using TPixelFunction = glm::vec4(PixelProgram* d, const ProgramDataPack& args);
using PixelFunction = TPixelFunction*;

class PixelProgram
{
public:
    virtual void UseLights(const std::vector<Light*>& lights) noexcept {};
    virtual std::vector<VertexDataDescriptor> GetInputDefinition() const noexcept = 0;
    virtual PixelFunction GetEntry() const noexcept = 0;
    virtual ~PixelProgram() {}
};
