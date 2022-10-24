#pragma once
#include "model/vertex.h"
#include "data_pack.h"

struct VertexShaderOutputDispatchable : pro::dispatch<ProgramDataPack(const ProgramDataPack& args)>
{
    template <class TSelf>
    ProgramDataPack operator()(const TSelf& self, const ProgramDataPack& args) const noexcept
    {
        return self.GetOutput(args);
    }
};

struct VertexShaderInputDefinitionDispatchable : pro::dispatch<const std::vector<VertexDataDescriptor>&()>
{
    template <class TSelf>
    const std::vector<VertexDataDescriptor>& operator()(const TSelf& self) const noexcept
    {
        return self.GetInputDefinition();
    }
};

struct VertexShaderOutputDefinitionDispatchable : pro::dispatch<const std::vector<VertexDataDescriptor>&()>
{
    template <class TSelf>
    const std::vector<VertexDataDescriptor>& operator()(const TSelf& self) const noexcept
    {
        return self.GetOutputDefinition();
    }
};

struct VertexShaderFacade : pro::facade<
    VertexShaderInputDefinitionDispatchable,
    VertexShaderOutputDefinitionDispatchable,
    VertexShaderOutputDispatchable>
{

};
