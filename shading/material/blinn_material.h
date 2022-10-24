#pragma once
#include "shading/pixel_program.h"

class BlinnMaterial
{
    std::vector<LightEntry> lightEntry;
    std::uint8_t *textureData = nullptr;
    int textureW = 0;
    int textureH = 0;
    glm::vec3 cameraPos{0.0};
    int shininess = 32;
    float specularStrength = 1.0;

public:
    void SetDiffuseMap(std::uint8_t *data, int w, int h) noexcept
    {
        textureData = data;
        textureW = w, textureH = h;
    }
    void SetLightParameters(int shininess, float strength) noexcept
    {
        this->shininess = shininess;
        this->specularStrength = strength;
    }
    void SetViewPosition(const glm::vec3 &pos) noexcept { cameraPos = pos; }
    std::vector<VertexDataDescriptor> GetInputDefinition() const noexcept;
    void UseLights(const std::vector<Light *> &lights) noexcept;
    glm::vec4 GetPixelColor(const ProgramDataPack& args) const noexcept;
};
