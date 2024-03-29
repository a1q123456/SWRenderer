#pragma once
#include <glm/glm.hpp>

class Light;
using TLightFunction = glm::vec4(Light*, const glm::vec4 &);
using LightFunction = TLightFunction *;

using TLightIntensityFunction = float(Light*, const glm::vec4 &);
using LightIntensityFunction = TLightIntensityFunction *;

class Light
{
public:
    virtual bool IsAmbient() const noexcept { return false; }
    virtual float GetAmbientIntensity() const noexcept { return 0.0; }
    virtual LightFunction GetDirectionEntry() const noexcept = 0;
    virtual LightIntensityFunction GetIntensityEntry() const noexcept = 0;
    virtual LightFunction GetColorEntry() const noexcept = 0;
};

struct LightEntry
{
    LightFunction directionEntry = nullptr;
    LightFunction colorEntry = nullptr;
    LightIntensityFunction intensityEntry = nullptr;

    Light* self = nullptr;
    bool isAmbient = false;
    float ambientIntensity = 0.0;
};
