#include "pbr_material.h"
#include <cmath>
#include <cuda.h>

std::vector<VertexDataDescriptor> PBRMaterial::GetInputDefinition() const noexcept
{
    return {{VertexAttributes::TextureCoordinate, VertexAttributeTypes::Vec3},
            {VertexAttributes::Normal, VertexAttributeTypes::Vec3}};
}

__device__ glm::vec4 PBRMaterial::GetPixelColor(const ProgramDataPack& args) const noexcept
{
    auto uvw = args.GetData<glm::vec3>(0, VertexAttributes::TextureCoordinate);
    auto normal = args.GetData<glm::vec3>(0, VertexAttributes::Normal);

    auto imgX = glm::clamp((int)std::round(uvw.x * textureW), 0, textureW - 1);
    auto imgY = glm::clamp((int)std::round(uvw.y * textureH), 0, textureH - 1);
    glm::vec4 outColor{
        textureData[(imgY) * textureW * 4 + imgX * 4 + 0] / 255.0,
        textureData[(imgY) * textureW * 4 + imgX * 4 + 1] / 255.0,
        textureData[(imgY) * textureW * 4 + imgX * 4 + 2] / 255.0,
        1.0};

    return glm::vec4{glm::vec3{outColor}, 1.0};
}

using PixelColorFunction = glm::vec4 (PBRMaterial::*)(const ProgramDataPack &args) const;

__device__ PixelColorFunction getPixelColor = &PBRMaterial::GetPixelColor;

PixelFunction PBRMaterial::GetEntry() const noexcept
{
    PixelFunction ret = nullptr;
    cudaMemcpyFromSymbol(&ret, getPixelColor, sizeof(PixelColorFunction));
    return ret;
}
