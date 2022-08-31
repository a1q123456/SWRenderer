#pragma once
#include "exception/throw.h"
#include "texture-filtering/filters.h"


template<typename T, glm::length_t NChannels, glm::qualifier Q, std::size_t Dim>
struct SamplerDispatchable : pro::dispatch<
        glm::vec<NChannels, T, Q>(const TextureCoordinate<Dim + 1>&, const std::vector<ResourceView<Dim>>&)
    > {
    template <class TSelf>
    glm::vec<NChannels, T, Q> operator()(
        const TSelf& self, 
        const TextureCoordinate<Dim + 1>& coord, 
        const std::vector<ResourceView<Dim>>& resourceViews)
    {
        return self.template Sample<T, NChannels, Q, Dim>(coord, resourceViews);
    }
};

template<typename T, glm::length_t NChannels, glm::qualifier Q, std::size_t Dim>
struct SamplerFacade : pro::facade<SamplerDispatchable<T, NChannels, Q, Dim>> {};

template<typename T, glm::length_t NChannels, glm::qualifier Q, std::size_t Dim>
pro::proxy<SamplerFacade<T, NChannels, Q, Dim>> CreateSampler(ETextureFilteringMethods algo)
{
    switch (algo)
    {
    case ETextureFilteringMethods::Linear:
        return pro::make_proxy<SamplerFacade<T, NChannels, Q, Dim>>(LinearSamplerAlgorithm{});
    case ETextureFilteringMethods::Nearest:
        return pro::make_proxy<SamplerFacade<T, NChannels, Q, Dim>>(NearestSamplerAlgorithm{});
    case ETextureFilteringMethods::Cubic:
        return pro::make_proxy<SamplerFacade<T, NChannels, Q, Dim>>(CubicSamplerAlgorithm{});
    }
    ThrowException(SWRErrorCode::InvalidEnum);
}
