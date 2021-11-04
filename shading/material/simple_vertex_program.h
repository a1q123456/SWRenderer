#pragma once
#include "shading/vertex_program.h"

class SimpleVertexProgram : public VertexProgram
{
private:
    glm::mat4 modelTransform;
    glm::mat4 projVP;
public:
    void SetModelMatrix(const glm::mat4 &modelTransform) noexcept { this->modelTransform = modelTransform;}
    void SetViewProjectMatrix(const glm::mat4 &projVP) noexcept { this->projVP = projVP; }
    const std::vector<VertexDataDescriptor>& GetInput() const noexcept override;
    const std::vector<VertexDataDescriptor>& GetOutput() const noexcept override;
    VertexFunction GetEntry() const noexcept override;
};
