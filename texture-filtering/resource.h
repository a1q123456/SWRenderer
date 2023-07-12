#pragma once
#include "texture-filtering/resource_data_type.h"
#include <algorithm>
#include <numeric>

template<std::size_t N>
using TextureBoundary = glm::vec<N, float, glm::highp>;

using Texture1DBoundary = TextureBoundary<1>;
using Texture2DBoundary = TextureBoundary<2>;
using Texture3DBoundary = TextureBoundary<3>;


constexpr std::size_t sizeOf(EResourceDataType dataType);
template<EResourceDataType dataType>
constexpr std::size_t sizeOf();

template<std::size_t Dim>
using ResourceCoordinate = TextureBoundary<Dim>;

using ResourceCoordinate1D = ResourceCoordinate<1>;
using ResourceCoordinate2D = ResourceCoordinate<2>;
using ResourceCoordinate3D = ResourceCoordinate<3>;

template<typename TAllocator>
class BasicResource
{
    template<std::size_t Dim>
    friend class ResourceView;
private:
    std::uint8_t* data = nullptr;
    std::size_t sizeInBytes = 0;
    TAllocator allocator;
    bool ownData = false;

public:
    BasicResource() = default;
    BasicResource(const BasicResource&) = delete;
    BasicResource(BasicResource&&);
    BasicResource(const std::uint8_t* data, std::size_t sizeInBytes, TAllocator allocator = {});
    BasicResource(std::size_t sizeInBytes, TAllocator allocator = {});
    static BasicResource Attach(std::uint8_t* data, std::size_t sizeInBytes);

    BasicResource& operator=(const BasicResource&) = delete;
    BasicResource& operator=(BasicResource&&);

    const std::uint8_t* Data() const
    {
        return data;
    }

    std::uint8_t* Data()
    {
        return data;
    }

    ~BasicResource()
    {
        if (!ownData)
        {
            return;
        }
        allocator.deallocate(data, sizeInBytes);
    }
};

using Resource = BasicResource<std::allocator<std::uint8_t>>;

template<std::size_t Dim>
class ResourceView
{
    static_assert(Dim >= 1, "Dimenssion must be greater than or equal to 1");
private:
    Resource* resource = nullptr;
    EResourceDataType dataType = EResourceDataType::UInt8;
    std::size_t lineSize = 0;
    TextureBoundary<Dim> boundary;
    std::size_t start = 0;
    int channels = 0;

    /**
     * @brief Get the a element
     * @tparam T element type
     * @tparam NChannels number of channels
     * @tparam Q qualifier
     * @param offset index of the element
     * @return T& 
     */
    template<typename T, glm::length_t NChannels, glm::qualifier Q>
    glm::vec<NChannels, T, Q> GetItem(std::size_t offset) const noexcept;

    template<typename T, glm::length_t NChannels, glm::qualifier Q>
    void SetItem(std::size_t offset, const glm::vec<NChannels, T, Q>& val) noexcept;
public:
    ResourceView() = default;
    ResourceView(Resource* resource, std::size_t start, EResourceDataType dataType, int channels, std::size_t lineSize, const TextureBoundary<Dim>& boundary);

    void Rebind(Resource* newResource);

    template<typename T, glm::length_t NChannels, glm::qualifier Q>
    glm::vec<NChannels, T, Q> Get(const ResourceCoordinate<Dim>& coord) const
    {
        std::uint32_t offset = 0;
        if constexpr (Dim > 1)
        {
            offset = coord[1];
            for (int i = 2; i < Dim; i++)
            {
                offset *= coord[i];
            }
            offset *= lineSize;
        }
        offset += coord[0] * sizeOf(dataType) * channels;
        return GetItem<T, NChannels, Q>(offset);
    }

    template<typename T, glm::length_t NChannels, glm::qualifier Q>
    void Set(const ResourceCoordinate<Dim>& coord, const glm::vec<NChannels, T, Q>& val)
    {
        std::uint32_t offset = 0;
        if constexpr (Dim > 1)
        {
            offset = coord[1];
            for (int i = 2; i < Dim; i++)
            {
                offset *= coord[i];
            }
            offset *= lineSize;
        }
        offset += coord[0] * sizeOf(dataType) * channels;
        SetItem<T, NChannels, Q>(offset, val);
    }

    TextureBoundary<Dim> Boundary() const noexcept
    {
        return boundary;
    }

    std::uint8_t* Data() const noexcept
    {
        return resource->Data();
    }

    std::size_t SizeInBytes() const noexcept
    {
        return std::accumulate(std::begin(boundary), std::end(boundary), 0) * sizeOf(dataType);
    }

    EResourceDataType DataType() const noexcept
    {
        return dataType;
    }
};

#include "resource.inl"

