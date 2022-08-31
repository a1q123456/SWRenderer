#include "exception/throw.h"


#define _resource_impl_convertArray(pSrc, pDst, size, TSrc, TDst) \
{ \
    for (int i = 0; i < size; i++) \
    { \
        pDst[i] = reinterpret_cast<const TSrc*>(pSrc)[i] / static_cast<TDst>(std::numeric_limits<TSrc>::max()); \
    } \
}

#define _resource_impl_convertVector(pSrc, pDst, size, TSrc, TDst) \
{ \
    for (int i = 0; i < size; i++) \
    { \
        pDst[i] = std::clamp<TDst>(pSrc[i] * static_cast<TSrc>(std::numeric_limits<TDst>::max()), \
            std::numeric_limits<TDst>::min(),  \
            std::numeric_limits<TDst>::max() \
        ); \
    } \
}

#define _resource_impl_convertVector_unsinged(pSrc, pDst, size, TSrc, TDst) \
{ \
    for (int i = 0; i < size; i++) \
    { \
        auto val = pSrc[i] * static_cast<TSrc>(std::numeric_limits<TDst>::max());\
        pDst[i] = std::clamp<TDst>(val < 0 ? 0 : val, \
            std::numeric_limits<TDst>::min(),  \
            std::numeric_limits<TDst>::max() \
        ); \
    } \
}

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
    
    constexpr const std::size_t dataTypeSize[static_cast<int>(EResourceDataType::DataTypesCount)]
    {
        sizeof(float),
        sizeof(double),
        sizeof(std::int8_t),
        sizeof(std::uint8_t),
        sizeof(std::int8_t),
        sizeof(std::uint8_t),
        sizeof(std::int8_t),
        sizeof(std::uint8_t),
        sizeof(std::int8_t),
        sizeof(std::uint8_t)
    };

    template<typename T>
    void ConvertToVector(EResourceDataType dataType, const std::uint8_t* pData, int nChannels, T* out) noexcept
    {
        switch (dataType)
        {
        case EResourceDataType::UInt8:
            _resource_impl_convertArray(pData, out, nChannels, std::uint8_t, T);
            return;
        case EResourceDataType::UInt16:
            _resource_impl_convertArray(pData, out, nChannels, std::uint16_t, T);
            return;
        case EResourceDataType::UInt32:
            _resource_impl_convertArray(pData, out, nChannels, std::uint32_t, T);
            return;
        case EResourceDataType::UInt64:
            _resource_impl_convertArray(pData, out, nChannels, std::uint64_t, T);
            return;
        case EResourceDataType::Int8:
            _resource_impl_convertArray(pData, out, nChannels, std::int8_t, T);
            return;
        case EResourceDataType::Int16:
            _resource_impl_convertArray(pData, out, nChannels, std::int16_t, T);
            return;
        case EResourceDataType::Int32:
            _resource_impl_convertArray(pData, out, nChannels, std::int32_t, T);
            return;
        case EResourceDataType::Int64:
            _resource_impl_convertArray(pData, out, nChannels, std::int64_t, T);
            return;
        case EResourceDataType::Float:
            _resource_impl_convertArray(pData, out, nChannels, float, T);
            return;
        case EResourceDataType::Double:
            _resource_impl_convertArray(pData, out, nChannels, double, T);
            return;
        }
        return;
    }

    template<typename T>
    void ConvertToMemory(EResourceDataType dataType, const T* pSrc, int nChannels, std::uint8_t* pDst) noexcept
    {
        switch (dataType)
        {
        case EResourceDataType::UInt8: 
            _resource_impl_convertVector_unsinged(pSrc, pDst, nChannels, T, std::uint8_t);
            return;
        case EResourceDataType::UInt16:
            _resource_impl_convertVector_unsinged(pSrc, pDst, nChannels, T, std::uint16_t);
            return;
        case EResourceDataType::UInt32:
            _resource_impl_convertVector_unsinged(pSrc, pDst, nChannels, T, std::uint32_t);
            return;
        case EResourceDataType::UInt64:
            _resource_impl_convertVector_unsinged(pSrc, pDst, nChannels, T, std::uint64_t);
            return;
        case EResourceDataType::Int8:
            _resource_impl_convertVector(pSrc, pDst, nChannels, T, std::int8_t);
            return;
        case EResourceDataType::Int16:
            _resource_impl_convertVector(pSrc, pDst, nChannels, T, std::int16_t);
            return;
        case EResourceDataType::Int32:
            _resource_impl_convertVector(pSrc, pDst, nChannels, T, std::int32_t);
            return;
        case EResourceDataType::Int64:
            _resource_impl_convertVector(pSrc, pDst, nChannels, T, std::int64_t);
            return;
        case EResourceDataType::Float:
            _resource_impl_convertVector(pSrc, pDst, nChannels, T, float);
            return;
        case EResourceDataType::Double:
            _resource_impl_convertVector(pSrc, pDst, nChannels, T, double);
            return;
        }
        return;
    }
}

template<EResourceDataType dataType>
constexpr inline std::size_t sizeOf()
{
    return _resource_impl::SizeInfo<dataType>::value;
}

constexpr inline std::size_t sizeOf(EResourceDataType dataType)
{
    return _resource_impl::dataTypeSize[static_cast<int>(dataType)];
}

template<typename TAllocator>
BasicResource<TAllocator>::BasicResource(BasicResource&& resource)
{
    if (resource.ownData && allocator != resource.allocator)
    {
        allocator = std::move(resource.allocator);
    }
    ownData = resource.ownData;
    data = resource.data;
    sizeInBytes = resource.sizeInBytes;
    resource.data = nullptr;
    resource.sizeInBytes = 0;
}

template<typename TAllocator>
BasicResource<TAllocator>& BasicResource<TAllocator>::operator=(BasicResource&& resource)
{
    if (resource.ownData && allocator != resource.allocator)
    {
        allocator = std::move(resource.allocator);
    }
    ownData = resource.ownData;
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
    TAllocator allocator): 
    sizeInBytes(sizeInBytes), 
    allocator(std::move(allocator))
{
    data = this->allocator.allocate(sizeInBytes);
    ownData = true;
}

template<typename TAllocator>
BasicResource<TAllocator> BasicResource<TAllocator>::Attach(std::uint8_t* data, std::size_t sizeInBytes)
{
    BasicResource<TAllocator> ret;
    ret.ownData = false;
    ret.data = data;
    ret.sizeInBytes = sizeInBytes;
    return ret;
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
glm::vec<NChannels, T, Q> ResourceView<Dim>::GetItem(std::size_t offset) const noexcept
{
    auto minChannels = std::min(NChannels, channels);
    glm::vec<NChannels, T, Q> ret{0};
    auto pDst = &ret.x;
    auto pSrc = resource->data + start + offset;
    _resource_impl::ConvertToVector<T>(dataType, pSrc, minChannels, pDst);
    return ret;
}

template<std::size_t Dim>
template<typename T, glm::length_t NChannels, glm::qualifier Q>
void ResourceView<Dim>::SetItem(std::size_t offset, const glm::vec<NChannels, T, Q>& val) noexcept
{
    auto minChannels = std::min(NChannels, channels);
    auto pSrc = &val.x;
    auto pDst = resource->data + start + offset;
    _resource_impl::ConvertToMemory<T>(dataType, pSrc, minChannels, pDst);
    return;
}
