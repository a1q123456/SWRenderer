#pragma once

template<std::size_t N>
using TextureBoundary = glm::vec<N, float, glm::highp>;

using Texture1DBoundary = TextureBoundary<1>;
using Texture2DBoundary = TextureBoundary<2>;
using Texture3DBoundary = TextureBoundary<3>;

enum class EResourceDataType
{
    Float = 0,
    Double,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64
};

std::size_t sizeOf(EResourceDataType dataType);
template<EResourceDataType dataType>
std::size_t sizeOf();

template<std::size_t Dim>
using ResourceCoordinate = TextureBoundary<Dim>;

template<typename TAllocator>
class BasicResource
{
    template<std::size_t Dim>
    friend class ResourceView;
private:
    std::uint8_t* data;
    std::size_t sizeInBytes = 0;
    TAllocator allocator;

public:
    BasicResource() = default;
    BasicResource(const BasicResource&) = delete;
    BasicResource(BasicResource&&);
    BasicResource(const std::uint8_t* data, std::size_t sizeInBytes, TAllocator allocator = {});
    BasicResource(std::size_t sizeInBytes, TAllocator allocator = {});

    BasicResource& operator=(const BasicResource&) = delete;
    BasicResource& operator=(BasicResource&&);

    std::uint8_t* Data() const
    {
        return data;
    }

    ~BasicResource()
    {
        if (data == nullptr)
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
    EResourceDataType dataType = EResourceDataType::UInt;
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
    glm::vec<NChannels, T, Q> GetItem(std::size_t offset) const;

    template<typename T>
    T ConvertType(std::uint8_t* pData) const noexcept;

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

    TextureBoundary<Dim> Boundary() const noexcept
    {
        return boundary;
    }

    std::uint8_t* Data() const noexcept
    {
        return resource.data.get();
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
