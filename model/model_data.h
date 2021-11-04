#pragma once
#include "vertex.h"
#include "data_pack.h"

class ModelData
{
private:
    std::vector<int> indexData;
    ModelDataPack dataPack;

public:
    size_t GetNumberIndices() const noexcept { return indexData.size(); }
    void SetIndexList(const std::vector<int> &indexData) noexcept;
    void SetVertexList(const std::vector<float> &vertexData) noexcept;
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
