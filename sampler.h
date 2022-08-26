#pragma once
#include "exception/throw.h"
#include "texture-filtering/filters.h"

template<typename ...TAlgo>
class SamplerAlgorithmFactory
{
private:
    TextureFilteringMethods samplerAlgorithmID = TextureFilteringMethods::Linear;

    template<typename T, glm::length_t NChannels, glm::qualifier Q, std::size_t Dim, typename Algo>
    glm::vec<NChannels, T, Q> Sample(
        const TextureCoordinate<Dim>& coord, 
        const std::vector<ResourceView<Dim>>& resourceViews,
        double distance, 
        Algo)
    {
        if (samplerAlgorithmID == Algo::ID)
        {
            return Algo::template Sample<T, NChannels, Q, Dim>(coord, resourceViews, distance);
        }
        ThrowException(SWRErrorCode::IndexOutOfRange);
    }

    template<typename T, glm::length_t NChannels, glm::qualifier Q, std::size_t Dim, typename Algo, typename ...Rest>
    glm::vec<NChannels, T, Q> Sample(
        const TextureCoordinate<Dim>& coord, 
        const std::vector<ResourceView<Dim>>& resourceViews,
        double distance, 
        Algo, Rest...)
    {
        if (samplerAlgorithmID == Algo::ID)
        {
            return Algo::template Sample<T, NChannels, Q, Dim>(coord, resourceViews, distance);
        }
        return Sample<T, NChannels, Q, Dim, Rest...>(coord, resourceViews, distance, Rest{}...);
    }

public:
    SamplerAlgorithmFactory() = default;
    SamplerAlgorithmFactory(TextureFilteringMethods method) : samplerAlgorithmID(method) {}

    template<typename T, glm::length_t NChannels, glm::qualifier Q, std::size_t Dim>
    glm::vec<NChannels, T, Q> Sample(
        const TextureCoordinate<Dim>& coord, 
        const std::vector<ResourceView<Dim>>& resourceViews,
        double distance) 
    {
        return Sample<T, NChannels, Q, Dim, TAlgo...>(coord, resourceViews, distance, TAlgo{}...);
    }
};

using SamplerAlgorithms = SamplerAlgorithmFactory<
    LinearSamplerAlgorithm,
    NearestSamplerAlgorithm,
    CubicSamplerAlgorithm>;
