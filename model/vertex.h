#pragma once

enum class VertexAttributes: std::uint32_t
{
    Position = 1,
    TextureCoordinate = 2,
    Normal = 4,
    Color = 8,
    Custom = 0
};

enum class VertexAttributeTypes: size_t
{
    Float = 1,
    Vec2 = 2,
    Vec3 = 3,
    Vec4 = 4
};

struct VertexDataDescriptor
{
    VertexAttributes attr;
    VertexAttributeTypes type;
    const char* name = nullptr;;
};

template<typename T>
inline std::uint32_t getVertexAttributeMask(T&& descriptors)
{
    std::uint32_t ret = 0;
    for (auto&& d : descriptors)
    {
        ret |= (std::uint32_t)d.attr;
    }
    return ret;
}
