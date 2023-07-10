#pragma once
#include "model/vertex.h"
#include "shading/light/light.h"
#include "data_pack.h"

struct PixelShaderInputDefinitionDispatchable : pro::dispatch<std::vector<VertexDataDescriptor>()>
{
    template <class TSelf>
    std::vector<VertexDataDescriptor> operator()(const TSelf& self) const noexcept
    {
        return self.GetInputDefinition();
    }
};

struct PixelShaderOutputColorDispatchable : pro::dispatch<glm::vec4(const ProgramDataPack& args)>
{
    template <class TSelf>
    glm::vec4 operator()(
        const TSelf& self,
        const ProgramDataPack& args) const noexcept
    {
        return self.GetPixelColor(args);
    }
};

struct PixelShaderFacade : pro::facade<
    PixelShaderInputDefinitionDispatchable,
    PixelShaderOutputColorDispatchable>
{

};
