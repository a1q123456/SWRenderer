#include "exception/throw.h"

namespace _resource_impl
{
    template<EResourceDataType dataType>
    struct SizeInfo { };
    
    template<>
    struct SizeInfo<EResourceDataType::Float> { static constexpr std::size_t value = sizeof(float); };
    
    template<>
    struct SizeInfo<EResourceDataType::Double> { static constexpr std::size_t value = sizeof(double); };
    
    template<>
    struct SizeInfo<EResourceDataType::Int8> { static constexpr std::size_t value = sizeof(std::int8_t); };
    
    template<>
    struct SizeInfo<EResourceDataType::UInt8> { static constexpr std::size_t value = sizeof(std::uint8_t); };
    
    template<>
    struct SizeInfo<EResourceDataType::Int16> { static constexpr std::size_t value = sizeof(std::int16_t); };
    
    template<>
    struct SizeInfo<EResourceDataType::UInt16> { static constexpr std::size_t value = sizeof(std::uint16_t); };
    
    template<>
    struct SizeInfo<EResourceDataType::Int32> { static constexpr std::size_t value = sizeof(std::int32_t); };
    
    template<>
    struct SizeInfo<EResourceDataType::UInt32> { static constexpr std::size_t value = sizeof(std::uint32_t); };
    
    template<>
    struct SizeInfo<EResourceDataType::Int64> { static constexpr std::size_t value = sizeof(std::int64_t); };
    
    template<>
    struct SizeInfo<EResourceDataType::UInt64> { static constexpr std::size_t value = sizeof(std::uint64_t); };
}

template<EResourceDataType dataType>
inline std::size_t sizeOf()
{
    return _resource_impl::SizeInfo<dataType>::value;
}

inline std::size_t sizeOf(EResourceDataType dataType)
{
    static std::unordered_map<EResourceDataType, std::size_t> dataTypeSize
    {
        { EResourceDataType::Float, sizeof(float) },
        { EResourceDataType::Double, sizeof(double) },
        { EResourceDataType::Int8, sizeof(std::int8_t) },
        { EResourceDataType::UInt8, sizeof(std::uint8_t) },
        { EResourceDataType::Int16, sizeof(std::int8_t) },
        { EResourceDataType::UInt16, sizeof(std::uint8_t) },
        { EResourceDataType::Int32, sizeof(std::int8_t) },
        { EResourceDataType::UInt32, sizeof(std::uint8_t) },
        { EResourceDataType::Int64, sizeof(std::int8_t) },
        { EResourceDataType::UInt64, sizeof(std::uint8_t) },
    };

    return dataTypeSize.at(dataType);
}

template<typename TAllocator>
BasicResource<TAllocator>::BasicResource(BasicResource&& resource) :
    allocator(std::move(resource.allocator))
{
    data = resource.data;
    sizeInBytes = resource.sizeInBytes;
    resource.data = nullptr;
    resource.sizeInBytes = 0;
}

template<typename TAllocator>
BasicResource<TAllocator>& BasicResource<TAllocator>::operator=(BasicResource&& resource)
{
    allocator = std::move(resource.allocator);
    data = resource.data;
    resource.data = nullptr;
    sizeInBytes = resource.sizeInBytes;
    resource.sizeInBytes = 0;
    return *this;
}

template<typename TAllocator>
BasicResource<TAllocator>::BasicResource(
    const std::uint8_t* data, 
    std::size_t sizeInBytes,
    TAllocator allocator) : BasicResource(sizeInBytes, std::move(allocator))
{
    std::memcpy(this->data, data, sizeInBytes);
}

template<typename TAllocator>
BasicResource<TAllocator>::BasicResource(
    std::size_t sizeInBytes,
    TAllocator allocator): sizeInBytes(sizeInBytes), allocator(std::move(allocator))
{
    this->data = this->allocator.allocate(sizeInBytes);
}

template<std::size_t Dim>
ResourceView<Dim>::ResourceView(
    Resource* res, 
    std::size_t start,
    EResourceDataType dataType,
    int channels, 
    std::size_t lineSize,
    const TextureBoundary<Dim>& boundary) :
    resource(res),
    dataType(dataType),
    channels(channels),
    lineSize(lineSize),
    start(start),
    boundary(boundary)
{
    if (resource->sizeInBytes % sizeOf(dataType) != 0 ||
        start % sizeOf(dataType) != 0)
    {
        ThrowException(SWRErrorCode::MisalignedMemoryAccess);
    }
}


template<std::size_t Dim>
void ResourceView<Dim>::Rebind(Resource* newResource)
{
    resource = newResource;
}

template<std::size_t Dim>
template<typename T, glm::length_t NChannels, glm::qualifier Q>
glm::vec<NChannels, T, Q> ResourceView<Dim>::GetItem(std::size_t offset) const
{
#if defined(DEBUG)
    if (offset >= resource->sizeInBytes)
    {
        ThrowException(SWRErrorCode::IndexOutOfRange);
    }
    if (offset % sizeOf(dataType) != 0)
    {
        ThrowException(SWRErrorCode::MisalignedMemoryAccess);
    }
#endif

    glm::vec<NChannels, T, Q> ret{0};
    for (int i = 0; i < std::min(NChannels, channels); i++)
    {
        ret[i] = ConvertType<T>((resource->data + offset + i));
    }

    return ret;
}

template<std::size_t Dim>
template<typename T>
T ResourceView<Dim>::ConvertType(std::uint8_t* pData) const noexcept
{
    switch (dataType)
    {
    case EResourceDataType::UInt8:
        return *pData;
    case EResourceDataType::UInt16:
        return *reinterpret_cast<std::uint16_t*>(pData);
    case EResourceDataType::UInt32:
        return *reinterpret_cast<std::uint32_t*>(pData);
    case EResourceDataType::UInt64:
        return *reinterpret_cast<std::uint64_t*>(pData);
    case EResourceDataType::Int8:
        return *reinterpret_cast<std::int8_t*>(pData);
    case EResourceDataType::Int16:
        return *reinterpret_cast<std::int16_t*>(pData);
    case EResourceDataType::Int32:
        return *reinterpret_cast<std::int32_t*>(pData);
    case EResourceDataType::Int64:
        return *reinterpret_cast<std::int64_t*>(pData);
    case EResourceDataType::Float:
        return *reinterpret_cast<float*>(pData);
    case EResourceDataType::Double:
        return *reinterpret_cast<double*>(pData);
    }
    return 0;
}
