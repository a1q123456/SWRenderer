#pragma once

class RayTracingRenderer
{
public:
    // void SetProgram(ProgramContext& programCtx);
    // void SetMesh(ModelData &mesh);
    // void SetCamera(const Camera& camera);
    void Render();
private:
    glm::mat4 viewMatrix;
};
