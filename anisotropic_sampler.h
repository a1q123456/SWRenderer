#pragma once
#include "sampler.h"

class AnisotropicSampler
{
public:
    void AttachResource(std::span<float> buffer, ResourceDescriptor rd);
    void AttachResource(const std::vector<std::span<float>>& arrayBuffer, ResourceDescriptor rd);
    glm::vec4 Sample(const glm::vec2& coord);
private:
    std::vector<std::span<float>> arrayBuffer;
    ResourceDescriptor rd;
    static constexpr int maxAniso = 4;
};

#include "anisotropic_sampler.inl"
