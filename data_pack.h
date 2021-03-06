#pragma once
#include "model/vertex.h"

template <typename TStorage>
class DataPack
{
    static constexpr size_t MAX_DESCRIPTORS = 20;

    TStorage storage;
    size_t elementSize = 0;

    std::array<VertexDataDescriptor, MAX_DESCRIPTORS> descriptors;
    std::array<std::pair<VertexAttributes, VertexAttributeTypes>, MAX_DESCRIPTORS> descriptorMap;
    std::array<std::pair<std::string_view, VertexAttributeTypes>, MAX_DESCRIPTORS> nameDescriptorMap;
    std::array<std::pair<VertexAttributes, size_t>, MAX_DESCRIPTORS> offsetMap;
    std::array<std::pair<std::string_view, size_t>, MAX_DESCRIPTORS> nameOffsetMap;

private:
    template <typename TMap, typename TKey, typename TVal = decltype(std::declval<TMap>()[0].second)>
    TVal MapGet(TMap &&map, TKey &&key) const
    {
        return std::find_if(std::begin(map), std::end(map), [&](auto &&pair)
                            { return pair.first == key; })
            ->second;
    }

    template <typename TMap, typename TKey, typename TVal = decltype(std::declval<TMap>()[0].second)>
    const TVal *MapFind(TMap &&map, TKey &&key) const
    {
        auto ret = std::find_if(std::begin(map), std::end(map), [&](auto &&pair)
                                { return pair.first == key; });
        if (ret == std::end(map))
        {
            return nullptr;
        }
        return &(ret->second);
    }

    template <typename T>
    void SetDataImpl(int start, size_t offset, T &&val)
    {
        storage[start + offset] = std::forward<T>(val);
    }

    template <typename T, typename... TRest>
    void SetDataImpl(int start, size_t offset, T &&val, TRest &&...rest)
    {
        storage[start + offset] = std::forward<T>(val);
        SetDataImpl(start, offset + 1, std::forward<TRest>(rest)...);
    }

public:
    DataPack() = default;
    template <typename T>
    DataPack(T &&dp) noexcept : storage(std::forward<T>(dp)) {}
    DataPack(const DataPack &other) noexcept : storage(other.storage),
                                               descriptors(other.descriptors),
                                               descriptorMap(other.descriptorMap),
                                               nameDescriptorMap(other.nameDescriptorMap),
                                               elementSize(other.elementSize),
                                               offsetMap(other.offsetMap),
                                               nameOffsetMap(other.nameOffsetMap)
    {
    }
    DataPack(DataPack &&other) noexcept
        : storage(std::move(other.storage)),
          descriptors(std::move(other.descriptors)),
          descriptorMap(std::move(other.descriptorMap)),
          nameDescriptorMap(std::move(other.nameDescriptorMap)),
          elementSize(std::move(other.elementSize)),
          offsetMap(std::move(other.offsetMap)),
          nameOffsetMap(std::move(other.nameOffsetMap))
    {
    }

    DataPack &operator=(const DataPack &other) noexcept
    {
        storage = other.storage;
        descriptors = other.descriptors;
        descriptorMap = other.descriptorMap;
        nameDescriptorMap = other.nameDescriptorMap;
        elementSize = other.elementSize;
        offsetMap = other.offsetMap;
        nameOffsetMap = other.nameOffsetMap;
        return *this;
    }
    DataPack &operator=(DataPack &&other) noexcept
    {
        storage = std::move(other.storage);
        descriptors = std::move(other.descriptors);
        descriptorMap = std::move(other.descriptorMap);
        nameDescriptorMap = std::move(other.nameDescriptorMap);
        elementSize = std::move(other.elementSize);
        offsetMap = std::move(other.offsetMap);
        nameOffsetMap = std::move(other.nameOffsetMap);
        return *this;
    }

    void SetDataDescriptor(const std::vector<VertexDataDescriptor> &descriptors) noexcept
    {
        std::copy(std::begin(descriptors), std::end(descriptors), std::begin(this->descriptors));
        for (int i = 0; i < descriptors.size(); i++)
        {
            auto &&d = descriptors[i];
            if (d.attr == VertexAttributes::Custom)
            {
                nameOffsetMap[i] = std::make_pair(std::string_view{d.name}, elementSize);
                nameDescriptorMap[i] = std::make_pair(std::string_view{d.name}, d.type);
            }
            offsetMap[i] = std::make_pair(d.attr, elementSize);
            descriptorMap[i] = std::make_pair(d.attr, d.type);

            elementSize += (size_t)d.type;
        }
    }

    std::vector<VertexAttributes> GetAttributes() const noexcept
    {
        std::vector<VertexAttributes> ret;
        std::transform(std::cbegin(descriptors), std::cend(descriptors), std::back_inserter(ret), [](auto &&d)
                       { return d.attr; });
        return ret;
    }

    std::uint32_t GetAttributeMask() const noexcept
    {
        return getVertexAttributeMask(descriptors);
    }

    VertexAttributeTypes GetType(VertexAttributes attr) const noexcept
    {
        return MapGet(descriptorMap, attr);
    }

    bool HasAttribute(VertexAttributes attr) const noexcept
    {
        return MapFind(descriptorMap, attr) != nullptr;
    }

    bool HasCustomAttribute(std::string_view name) const noexcept
    {
        return MapFind(nameDescriptorMap, name) != nullptr;
    }

    template <typename T>
    T GetData(int index, VertexAttributes attr) const noexcept
    {
        assert(attr != VertexAttributes::Custom);
        auto type = MapGet(descriptorMap, attr);
        auto start = elementSize * index + MapGet(offsetMap, attr);
        T ret;
        for (int i = 0; i < (int)type; i++)
        {
            ret[i] = storage[start + i];
        }
        return ret;
    }

    template <typename T>
    T GetData(int index, std::string_view name) const noexcept
    {
        auto type = MapGet(nameDescriptorMap, name);
        auto start = elementSize * index + MapGet(nameOffsetMap, name);
        T ret;
        for (int i = 0; i < (int)type; i++)
        {
            ret[i] = storage[start + i];
        }
        return ret;
    }

    float GetData(int index, int nItem, VertexAttributes attr) const noexcept
    {
        assert(attr != VertexAttributes::Custom);
        auto type = MapGet(descriptorMap, attr);
        auto start = elementSize * index + MapGet(offsetMap, attr);
        return storage[start + nItem];
    }

    float GetData(int index, int nItem, int dataIndex) const noexcept
    {
        auto type = descriptorMap[dataIndex].second;
        auto start = elementSize * index + offsetMap[dataIndex].second;
        return storage[start + nItem];
    }

    float GetData(int index, int nItem, std::string_view name) const noexcept
    {
        auto type = MapGet(nameDescriptorMap, name);
        auto start = elementSize * index + MapGet(nameOffsetMap, name);
        return storage[start + nItem];
    }

    void SetData(int index, int nItem, VertexAttributes attr, float val)
    {
        assert(attr != VertexAttributes::Custom);
        auto type = MapGet(descriptorMap, attr);
        auto start = elementSize * index + MapGet(offsetMap, attr);
        SetDataImpl(start, nItem, val);
    }

    void SetData(int index, int nItem, int dataIdx, float val)
    {
        auto type = descriptorMap[dataIdx].second;
        auto start = elementSize * index + offsetMap[dataIdx].second;
        SetDataImpl(start, nItem, val);
    }

    void SetData(int index, int nItem, std::string_view name, float val)
    {
        auto type = MapGet(nameDescriptorMap, name);
        auto start = elementSize * index + MapGet(nameOffsetMap, name);
        SetDataImpl(start, nItem, val);
    }

    template <typename T>
    void SetData(int index, VertexAttributes attr, T &&val)
    {
        assert(attr != VertexAttributes::Custom);
        auto type = MapGet(descriptorMap, attr);
        auto start = elementSize * index + MapGet(offsetMap, attr);
        assert((size_t)type == 1);
        SetDataImpl(start, 0, std::forward<T>(val));
    }

    template <typename... TRest>
    void SetData(int index, VertexAttributes attr, TRest &&...rest)
    {
        assert(attr != VertexAttributes::Custom);
        auto type = MapGet(descriptorMap, attr);
        auto start = elementSize * index + MapGet(offsetMap, attr);
        assert((size_t)type == sizeof...(rest));
        SetDataImpl(start, 0, std::forward<TRest>(rest)...);
    }

    template <typename T>
    void SetData(int index, std::string_view name, T &&val)
    {
        auto type = MapGet(nameDescriptorMap, name);
        auto start = elementSize * index + MapGet(nameOffsetMap, name);
        assert((size_t)type == 1);
        SetDataImpl(start, 0, std::forward<T>(val));
    }

    template <typename... TRest>
    void SetData(int index, std::string_view name, TRest &&...rest)
    {
        auto type = MapGet(nameDescriptorMap, name);
        auto start = elementSize * index + MapGet(nameOffsetMap, name);
        assert((size_t)type == sizeof...(rest));
        SetDataImpl(start, 0, std::forward<TRest>(rest)...);
    }
};

constexpr size_t MAX_PROGRAM_DATA_ITEMS = 40;

using ModelDataPack = DataPack<std::vector<float>>;
using ProgramDataPack = DataPack<std::array<float, MAX_PROGRAM_DATA_ITEMS>>;
