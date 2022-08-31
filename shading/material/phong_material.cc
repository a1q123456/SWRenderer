#include "phong_material.h"

std::vector<VertexDataDescriptor> PhongMaterial::GetInput() const noexcept
{
    return {{VertexAttributes::TextureCoordinate, VertexAttributeTypes::Vec3},
            {VertexAttributes::Normal, VertexAttributeTypes::Vec3},
            {VertexAttributes::Custom, VertexAttributeTypes::Vec3, "fragPos"}};
}

PixelFunction PhongMaterial::GetEntry() const noexcept
{
    return [](PixelProgram *d, const ProgramDataPack &args) -> glm::vec4
    {
        auto self = static_cast<PhongMaterial *>(d);
        auto uvw = args.GetData<glm::vec3>(0, VertexAttributes::TextureCoordinate);
        auto normal = args.GetData<glm::vec3>(0, VertexAttributes::Normal);
        auto fragPos = args.GetData<glm::vec3>(0, "fragPos");

        auto imgX = std::clamp((int)std::round(uvw.x * self->textureW), 0, self->textureW - 1);
        auto imgY = std::clamp((int)std::round(uvw.y * self->textureH), 0, self->textureH - 1) + 1;
        glm::vec4 outColor{
            self->textureData[(imgY)*self->textureW * 4 + imgX * 4 + 0] / 255.0,
            self->textureData[(imgY)*self->textureW * 4 + imgX * 4 + 1] / 255.0,
            self->textureData[(imgY)*self->textureW * 4 + imgX * 4 + 2] / 255.0,
            1.0};
        glm::vec4 fragPos4{fragPos, 1.0};
        glm::vec4 lightValue{0.0};
        for (auto &&light : self->lightEntry)
        {
            auto color = light.colorEntry(light.self, fragPos4);
            if (light.isAmbient)
            {

                lightValue += light.ambientIntensity * color;
                continue;
            }
            auto intensity = light.intensityEntry(light.self, fragPos4);

            auto lightDir = glm::normalize(glm::vec3{light.directionEntry(light.self, fragPos4)});
            auto normalDir = glm::normalize(normal);
            auto diffuse = std::clamp(glm::dot(lightDir, normalDir), 0.f, 1.f);

            auto cameraDir = glm::normalize(self->cameraPos - fragPos);
            auto reflectVec = glm::reflect(-lightDir, normalDir);
            auto specular = std::pow(std::clamp(glm::dot(reflectVec, cameraDir), 0.f, 1.f), self->shininess) * self->specularStrength;
            lightValue += (diffuse + (float)specular * color);
        }

        return glm::vec4{glm::vec3{outColor * lightValue}, 1.0};
    };
}

void PhongMaterial::UseLights(const std::vector<Light *> &lights) noexcept
{
    std::transform(std::cbegin(lights), std::cend(lights), std::back_inserter(lightEntry), [](auto &&l)
                   { return LightEntry{
                         l->GetDirectionEntry(),
                         l->GetColorEntry(),
                         l->GetIntensityEntry(),
                         l,
                         l->IsAmbient(),
                         l->GetAmbientIntensity()}; });
}
