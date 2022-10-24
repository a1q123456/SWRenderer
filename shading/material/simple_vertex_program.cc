#include "simple_vertex_program.h"

const std::vector<VertexDataDescriptor>& SimpleVertexProgram::GetInputDefinition() const noexcept
{
    static std::vector<VertexDataDescriptor> inputs{{VertexAttributes::Position, VertexAttributeTypes::Vec3},
                                                    {VertexAttributes::TextureCoordinate, VertexAttributeTypes::Vec3},
                                                    {VertexAttributes::Normal, VertexAttributeTypes::Vec3}};
    return inputs;
}

const std::vector<VertexDataDescriptor>& SimpleVertexProgram::GetOutputDefinition() const noexcept
{
    static std::vector<VertexDataDescriptor> outputs{{VertexAttributes::Position, VertexAttributeTypes::Vec3},
                                                     {VertexAttributes::TextureCoordinate, VertexAttributeTypes::Vec3},
                                                     {VertexAttributes::Normal, VertexAttributeTypes::Vec3},
                                                     {VertexAttributes::Custom, VertexAttributeTypes::Vec3, "fragPos"}};

    return outputs;
}

ProgramDataPack SimpleVertexProgram::GetOutput(const ProgramDataPack &args) const noexcept
{
    auto inPos = args.GetData<glm::vec3>(0, VertexAttributes::Position);
    auto inUv = args.GetData<glm::vec3>(0, VertexAttributes::TextureCoordinate);
    auto inNormal = args.GetData<glm::vec3>(0, VertexAttributes::Normal);

    auto mv = modelTransform * glm::vec4{inPos, 1};
    auto normal = glm::transpose(glm::inverse(modelTransform)) * glm::vec4{inNormal, 1};
    auto pv = projVP * mv;

    ProgramDataPack ret;
    ret.SetDataDescriptor(GetOutputDefinition());
    ret.SetData(0, VertexAttributes::Position, pv[0], pv[1], pv[2]);
    ret.SetData(0, VertexAttributes::TextureCoordinate, inUv[0], inUv[1], inUv[2]);
    ret.SetData(0, VertexAttributes::Normal, normal[0], normal[1], normal[2]);
    ret.SetData(0, "fragPos", mv[0], mv[1], mv[2]);
    return ret;
}
