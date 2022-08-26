#pragma once
#include "texture-filtering/filters.h"
#include "resource.h"


class NearestSamplerAlgorithm
{
public:
    static constexpr TextureFilteringMethods ID = TextureFilteringMethods::Nearest;

    template<typename T, glm::length_t NChannels, glm::qualifier Q, std::size_t Dim>
    static glm::vec<NChannels, T, Q> Sample(
        const TextureCoordinate<Dim>& coord, 
        const std::vector<ResourceView<Dim>>& resourceViews,
        double distance)
    {
        return resourceViews.front().template Get<T, NChannels, Q>(glm::floor(resourceViews.front().Boundary() * coord));
    }
};
