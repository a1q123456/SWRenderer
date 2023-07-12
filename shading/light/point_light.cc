#include "point_light.h"
#include <algorithm>

LightFunction PointLight::GetDirectionEntry() const noexcept
{
    return [](Light *d, const glm::vec4 &pos)
    {
        auto self = static_cast<PointLight *>(d);
        return self->pos - pos;
    };
}

LightIntensityFunction PointLight::GetIntensityEntry() const noexcept
{
    return [](Light *d, const glm::vec4 &pos)
    {
        auto self = static_cast<PointLight *>(d);
        return std::clamp(self->maxRange - glm::distance(self->pos, pos), 0.f, self->maxRange);
    };
}

LightFunction PointLight::GetColorEntry() const noexcept
{
    return [](Light *d, const glm::vec4 &pos)
    {
        auto self = static_cast<PointLight *>(d);
        return self->lightColor;
    };
}
