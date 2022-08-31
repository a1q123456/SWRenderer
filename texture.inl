#include "texture.h"
#include "image-processing/rescaling.h"

template<std::size_t N>
Texture<N>::Texture(Texture&& texture):
    resource(std::move(texture.resource)),
    resourceViews(std::move(texture.resourceViews)),
    desc(std::move(texture.desc)),
    filterMethod(filterMethod),
    boundaries(std::move(boundaries))
{
    texture.desc = {};
}

template<std::size_t N>
Texture<N>& Texture<N>::operator=(Texture&& texture)
{
    resource = std::move(texture.resource);
    resourceViews = std::move(texture.resourceViews);
    for (auto& rv : resourceViews)
    {
        rv.Rebind(&resource);
    }
    desc = std::move(texture.desc);
    filterMethod = texture.filterMethod;
    boundaries = std::move(boundaries);
    return *this;
}

template<std::size_t N>
Texture<N>::Texture(const std::vector<std::span<std::uint8_t>>& data, const TextureDesc<N>& desc, ETextureFilteringMethods filterMethod, int levelsToGenerate) :
    filterMethod(filterMethod)
{
    auto totalLevels = desc.levelsCount + levelsToGenerate;

    std::size_t size = 0;
    if constexpr (N > 1)
    {
        size = desc.boundary[1];
        for (int i = 2; i < N; i++)
        {
            size *= desc.boundary[i];
        }
        size *= desc.linesize;
    }

    std::vector<std::size_t> sizes;
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> lineSizes;
    std::vector<TextureBoundary<N>> boundaries;
    sizes.resize(totalLevels);
    offsets.resize(totalLevels);
    lineSizes.resize(totalLevels);
    boundaries.resize(totalLevels);
    offsets[0] = 0;
    sizes[0] = size;
    lineSizes[0] = desc.linesize;
    boundaries[0] = desc.boundary;
    std::size_t sumSize = size;

    auto den = std::pow(2, N);
    for (int i = 1; i < totalLevels; i++)
    {
        offsets[i] = sumSize;
        size /= den;
        sumSize += size;
        sizes[i] = size;
        lineSizes[i] = lineSizes[i - 1] / 2;
        
        boundaries[i] = boundaries[i - 1];
        boundaries[i] /= 2;
    }

    resource = Resource{sumSize};
    resourceViews.reserve(totalLevels);

    LoadMipmap(data, desc, sizes, offsets, lineSizes, boundaries);
    GenerateMipmap(data, desc, sizes, offsets, lineSizes, boundaries, desc.levelsCount - 1, levelsToGenerate);
    this->desc = desc;
    this->desc.levelsCount = totalLevels;
    this->boundaries = std::move(boundaries);
}

template<std::size_t N>
void Texture<N>::GenerateMipmap(
    const std::vector<std::span<std::uint8_t>>& data, 
    const TextureDesc<N>& desc, 
    const std::vector<std::size_t>& sizes, 
    const std::vector<std::size_t>& offsets,
    const std::vector<std::size_t>& lineSizes,
    const std::vector<TextureBoundary<N>>& boundaries,
    int fromLevel, 
    int toLevel)
{
    auto pixDesc = getPixelDesc(desc.pixelFormat);
    for (int i = fromLevel + 1; i < toLevel; i++)
    {
        if constexpr (N == 2)
        {
            rescaleImage2D(
                ETextureFilteringMethods::Cubic, 
                data[fromLevel], 
                desc.pixelFormat,
                boundaries[fromLevel].x,
                boundaries[fromLevel].y,
                lineSizes[fromLevel],
                boundaries[i].x,
                boundaries[i].y,
                lineSizes[i],
                std::span{resource.Data() + offsets[i], sizes[i]}
                );
        }
        resourceViews.emplace_back(&resource, offsets[i], desc.elementType, pixDesc->nbChannels, lineSizes[i], boundaries[i]);
    }
}

template<std::size_t N>
void Texture<N>::LoadMipmap(
    const std::vector<std::span<std::uint8_t>>& data, 
    const TextureDesc<N>& desc, 
    const std::vector<std::size_t>& sizes, 
    const std::vector<std::size_t>& offsets,
    const std::vector<std::size_t>& lineSizes,
    const std::vector<TextureBoundary<N>>& boundaries)
{
    auto pixDesc = getPixelDesc(desc.pixelFormat);
    for (int i = 0; i < desc.levelsCount; i++)
    {
        memcpy_s(resource.Data() + offsets[i], sizes[i], data[i].data(), data[i].size());
        resourceViews.emplace_back(&resource, offsets[i], desc.elementType, pixDesc->nbChannels, lineSizes[i], boundaries[i]);
    }
}

template<std::size_t N>
const TextureDesc<N> &Texture<N>::GetDesc() const noexcept
{
    return desc;
}

