#pragma once
#include "model/vertex.h"
#include "data_pack.h"

class SimpleVertexProgram
{
private:
    glm::mat4 modelTransform;
    glm::mat4 projVP;
public:
    void SetModelMatrix(const glm::mat4 &modelTransform) noexcept { this->modelTransform = modelTransform;}
    void SetViewProjectMatrix(const glm::mat4 &projVP) noexcept { this->projVP = projVP; }
    const std::vector<VertexDataDescriptor>& GetInputDefinition() const noexcept;
    const std::vector<VertexDataDescriptor>& GetOutputDefinition() const noexcept;
    ProgramDataPack GetOutput(const ProgramDataPack& args) const noexcept;
};
