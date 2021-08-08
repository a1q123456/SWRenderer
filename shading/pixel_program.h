#pragma once
#include "model/vertex.h"
#include "shading/light/light.h"
#include "data_pack.h"

class PixelProgram;
using TPixelFunction = glm::vec4(PixelProgram* d, const ProgramDataPack& args);
using PixelFunction = TPixelFunction*;

struct LightEntry
{
    LightFunction directionEntry = nullptr;
    LightFunction colorEntry = nullptr;
    LightIntensityFunction intensityEntry = nullptr;

    Light* self = nullptr;
    bool isAmbient = false;
    float ambientIntensity = 0.0;
};

class PixelProgram
{
public:
    virtual void UseLights(const std::vector<Light*>& lights) noexcept = 0;
    virtual std::vector<VertexDataDescriptor> GetInput() const noexcept = 0;
    virtual PixelFunction GetEntry() const noexcept = 0;
    virtual ~PixelProgram() {}
};
