#pragma once
#include "pixel_format.h"
#include "resource.h"
#include "sampler.h"

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
    SamplerAlgorithms samplerAlgorithm;
    std::vector<TextureBoundary<N>> boundaries;
    double distance = 0;
    
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
    Texture(const std::vector<std::span<std::uint8_t>>& data, const TextureDesc<N>& desc, TextureFilteringMethods filterMethod, int nbLevelToGenerate = -1);

    Texture& operator=(const Texture&) = delete;
    Texture& operator=(Texture&&);

    const TextureDesc<N>& GetDesc() const noexcept;

    template<typename T, glm::length_t NChannels, glm::qualifier Q = glm::defaultp>
    glm::vec<NChannels, T, Q> Sample(const TextureCoordinate<N>& coord)
    {
        return samplerAlgorithm.template Sample<T, NChannels, Q, N>(coord, resourceViews, distance);
    }

    void SetDistance(double distance)
    {
        this->distance = distance;
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
