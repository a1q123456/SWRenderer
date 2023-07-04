#pragma once
#include "vertex.h"
#include "data_pack.h"
#include "cuda/cuda_allocator.h"

template<template<typename> typename Allocator>
class BaseModelData
{
private:
    std::vector<int, Allocator<int>> indexData;
    ModelDataPack<Allocator> dataPack;

public:
    size_t GetNumberIndices() const noexcept { return indexData.size(); }
    void SetIndexList(const std::vector<int, Allocator<int>> &indexData) noexcept;
    void SetVertexList(const std::vector<float, Allocator<float>> &vertexData) noexcept;
    void SetVertexDescriptor(const std::vector<VertexDataDescriptor> &descriptors) noexcept;

    std::vector<VertexAttributes> GetAttributes() const noexcept;
    std::uint32_t GetAttributeMask() const noexcept;
    VertexAttributeTypes GetType(VertexAttributes attr) const noexcept;
    bool HasAttribute(VertexAttributes attr) const noexcept;

    template <typename T>
    T GetVertexData(int index, VertexAttributes attr) const noexcept
    {
        return dataPack.GetData<T>(indexData[index], attr);
    }

    float GetVertexData(int index, int nItem, VertexAttributes attr) const noexcept
    {
        return dataPack.GetData(indexData[index], nItem, attr);
    }
};

template<template<typename> typename Allocator>
void BaseModelData<Allocator>::SetIndexList(const std::vector<int, Allocator<int>> &indexData) noexcept
{
    this->indexData = indexData;
}

template<template<typename> typename Allocator>
void BaseModelData<Allocator>::SetVertexList(const std::vector<float, Allocator<float>> &vertexData) noexcept
{
    dataPack = ModelDataPack<Allocator>{vertexData};
}

template<template<typename> typename Allocator>
void BaseModelData<Allocator>::SetVertexDescriptor(const std::vector<VertexDataDescriptor> &descriptors) noexcept
{
    dataPack.SetDataDescriptor(descriptors);
}

template<template<typename> typename Allocator>
std::vector<VertexAttributes> BaseModelData<Allocator>::GetAttributes() const noexcept
{
    return dataPack.GetAttributes();
}

template<template<typename> typename Allocator>
std::uint32_t BaseModelData<Allocator>::GetAttributeMask() const noexcept
{
    return dataPack.GetAttributeMask();
}

template<template<typename> typename Allocator>
VertexAttributeTypes BaseModelData<Allocator>::GetType(VertexAttributes attr) const noexcept
{
    return dataPack.GetType(attr);
}

template<template<typename> typename Allocator>
bool BaseModelData<Allocator>::HasAttribute(VertexAttributes attr) const noexcept
{
    return dataPack.HasAttribute(attr);
}

using ModelData = BaseModelData<std::allocator>;
using CudaModelData = BaseModelData<CudaAllocator>;
