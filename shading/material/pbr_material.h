#pragma once
#include <glm/glm.hpp>
#include <vector>
#include "shading/light/light.h"
#include "model/vertex.h"
#include "data_pack.h"
#include "cuda_utils.h"

class PBRMaterial
{
    CudaPointer<std::uint8_t[]> textureData;
    int textureW = 0;
    int textureH = 0;

public:
    void SetDiffuseMap(CudaPointer<std::uint8_t[]> data, int w, int h) noexcept
    {
        textureData = std::move(data);
        textureW = w, textureH = h;
    }

    std::vector<VertexDataDescriptor> GetInputDefinition() const noexcept;
    glm::vec4 GetPixelColor(const ProgramDataPack& args) const noexcept;
};
