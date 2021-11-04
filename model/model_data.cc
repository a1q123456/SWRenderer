#include "model_data.h"

void ModelData::SetIndexList(const std::vector<int> &indexData) noexcept
{
    this->indexData = indexData;
}

void ModelData::SetVertexList(const std::vector<float> &vertexData) noexcept
{
    dataPack = ModelDataPack{vertexData};
}

void ModelData::SetVertexDescriptor(const std::vector<VertexDataDescriptor> &descriptors) noexcept
{
    dataPack.SetDataDescriptor(descriptors);
}

std::vector<VertexAttributes> ModelData::GetAttributes() const noexcept
{
    return dataPack.GetAttributes();
}

std::uint32_t ModelData::GetAttributeMask() const noexcept
{
    return dataPack.GetAttributeMask();
}

VertexAttributeTypes ModelData::GetType(VertexAttributes attr) const noexcept
{
    return dataPack.GetType(attr);
}

bool ModelData::HasAttribute(VertexAttributes attr) const noexcept
{
    return dataPack.HasAttribute(attr);
}

