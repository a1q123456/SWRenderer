#include "pbr_material.h"
#include <cmath>

std::vector<VertexDataDescriptor> PBRMaterial::GetInputDefinition() const noexcept
{
    return {{VertexAttributes::TextureCoordinate, VertexAttributeTypes::Vec3},
            {VertexAttributes::Normal, VertexAttributeTypes::Vec3}};
}

glm::vec4 PBRMaterial::GetPixelColor(const ProgramDataPack& args) const noexcept
{
    auto uvw = args.GetData<glm::vec3>(0, VertexAttributes::TextureCoordinate);
    auto normal = args.GetData<glm::vec3>(0, VertexAttributes::Normal);

    auto imgX = std::clamp((int)std::round(uvw.x * textureW), 0, textureW - 1);
    auto imgY = std::clamp((int)std::round(uvw.y * textureH), 0, textureH - 1);
    glm::vec4 outColor{
        textureData[(imgY) * textureW * 4 + imgX * 4 + 0] / 255.0,
        textureData[(imgY) * textureW * 4 + imgX * 4 + 1] / 255.0,
        textureData[(imgY) * textureW * 4 + imgX * 4 + 2] / 255.0,
        1.0};

    return glm::vec4{glm::vec3{outColor}, 1.0};
}
