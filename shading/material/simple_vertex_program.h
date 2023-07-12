#pragma once
#include "model/vertex.h"
#include "shading/vertex_program.h"
#include "model/data_pack.h"

class SimpleVertexProgram : public VertexProgram
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
    VertexFunction GetEntry() const noexcept 
    {
        return [](VertexProgram* d, const ProgramDataPack& args) 
        { 
            return static_cast<SimpleVertexProgram*>(d)->GetOutput(args);
        }; 
    }
};
