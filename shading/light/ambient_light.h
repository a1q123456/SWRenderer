#pragma once
#include "light.h"

class AmbientLight : public Light
{
    float intensity = 0.3;
    glm::vec4 color = {1, 1, 1, 1};

public:
    void SetParameters(float intensity, const glm::vec4 &color)
    {
        this->intensity = intensity;
        this->color = color;
    }

    bool IsAmbient() const noexcept override { return true; }
    float GetAmbientIntensity() const noexcept override { return intensity; }
    LightFunction GetDirectionEntry() const noexcept override { return nullptr; };
    LightIntensityFunction GetIntensityEntry() const noexcept override { return nullptr; };
    LightFunction GetColorEntry() const noexcept override
    {
        return [](Light *d, const glm::vec4 &)
        {
            auto self = static_cast<AmbientLight *>(d);
            return self->color;
        };
    }
};
