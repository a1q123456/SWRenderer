#pragma once
#include "image-processing/pixel_format.h"
#include "resource.h"
#include "texture-filtering/filters.h"
#include <span>
#include <cstdint>
#include <vector>

template<std::size_t N>
struct TextureDesc
{
    int levelsCount;
    TextureBoundary<N> boundary; 
    int linesize;
    EResourceDataType elementType;
    EPixelFormat pixelFormat;

    TextureDesc() = default;
    TextureDesc(int levelsCount, TextureBoundary<N> boundary, int linesize, EResourceDataType elementType, EPixelFormat pixelFormat):
        levelsCount(levelsCount),
        boundary(boundary),
        linesize(linesize),
        elementType(elementType),
        pixelFormat(pixelFormat)
    {}
};

using TextureDesc1D = TextureDesc<1>;
using TextureDesc2D = TextureDesc<2>;
using TextureDesc3D = TextureDesc<3>;


template<std::size_t N>
struct TexturePixelLocation
{
    int level;
    ResourceCoordinate<N> coordination;
};

template<std::size_t N>
class Texture
{
private:
    Resource resource;
    std::vector<ResourceView<N>> resourceViews;
    TextureDesc<N> desc;
    ETextureFilteringMethods filterMethod;
    std::vector<TextureBoundary<N>> boundaries;
    
    void LoadMipmap(
        const std::vector<std::span<std::uint8_t>>& data, 
        const TextureDesc<N>& desc, 
        const std::vector<std::size_t>& sizes, 
        const std::vector<std::size_t>& offsets,
        const std::vector<std::size_t>& lineSizes,
        const std::vector<TextureBoundary<N>>& boundaries);
    void GenerateMipmap(
        const std::vector<std::span<std::uint8_t>>& data, 
        const TextureDesc<N>& desc,
        const std::vector<std::size_t>& sizes, 
        const std::vector<std::size_t>& offsets,
        const std::vector<std::size_t>& lineSizes,
        const std::vector<TextureBoundary<N>>& boundaries,
        int fromLevel, 
        int toLevel);
public:
    Texture() = default;
    Texture(const Texture&) = delete;
    Texture(Texture&&);
    Texture(const std::vector<std::span<std::uint8_t>>& data, const TextureDesc<N>& desc, ETextureFilteringMethods filterMethod, int nbLevelToGenerate = -1);

    Texture& operator=(const Texture&) = delete;
    Texture& operator=(Texture&&);

    const TextureDesc<N>& GetDesc() const noexcept;

    template<typename T, glm::length_t NChannels, glm::qualifier Q = glm::defaultp>
    glm::vec<NChannels, T, Q> Sample(const TextureCoordinate<N + 1>& coord)
    {
        static auto sampler = CreateSampler<T, NChannels, Q, N>(filterMethod);
        return sampler.template invoke<SamplerDispatchable<T, NChannels, Q, N>>(coord, resourceViews);
    }

    template<typename T, glm::length_t NChannels, glm::qualifier Q = glm::defaultp>
    glm::vec<NChannels, T, Q> Sample(const TextureCoordinate<N>& coord)
    {
        static auto sampler = CreateSampler<T, NChannels, Q, N>(filterMethod);
        return sampler.template invoke<SamplerDispatchable<T, NChannels, Q, N>>(TextureCoordinate<N + 1>{coord, 0}, resourceViews);
    }

    template<typename T, glm::length_t NChannels, glm::qualifier Q = glm::defaultp>
    glm::vec<NChannels, T, Q> operator[](const TexturePixelLocation<N>& location) const
    {
        return glm::vec<NChannels, T, Q>{resourceViews[location.level][location.coordination]};
    }
};

using Texture1D = Texture<1>;
using Texture2D = Texture<2>;
using Texture3D = Texture<3>;

#include "texture.inl"
