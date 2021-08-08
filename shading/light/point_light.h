#pragma once
#include "light.h"

class PointLight : public Light
{
    glm::vec4 pos = glm::vec4{10, 10, 10, 1};
    glm::vec4 lightColor = glm::vec4{1, 1, 1, 1};
    float maxRange = 100.f;

public:
    bool IsAmbient() const noexcept { return false; }
    void SetRange(const float &range) noexcept { this->maxRange = range; }
    void SetPosition(const glm::vec4 &pos) noexcept { this->pos = pos; }
    void SetLightColor(const glm::vec4 &lightColor) noexcept { this->lightColor = lightColor; }
    LightFunction GetDirectionEntry() const noexcept override;
    LightIntensityFunction GetIntensityEntry() const noexcept override;
    LightFunction GetColorEntry() const noexcept override;
};
