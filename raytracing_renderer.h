#pragma once
#include "platform_defines.h"
#include "model/model_data.h"
#include "pixel_format.h"
#include "shading/vertex_program.h"
#include "shading/pixel_program.h"

class RayTracingProgramContext
{

};

class RayTracingRenderer
{
public:
    using ProgramContextType = RayTracingProgramContext;

    RayTracingRenderer(CanvasType&& canvas);

    void CreateBuffer(EPixelFormat pixelFormat);
    void SetProgram(ProgramContextType& programCtx);
    void SetMesh(ModelData &mesh);
    void ClearZBuffer();
    void ClearColorBuffer(std::uint32_t color);
    void Draw(float timeElapsed);
    CanvasType& Canvas();
    
    void SetMultiSampleLevel(int msaa);
    void ProjectionMatrix(glm::mat4x4 proj);

    static ProgramContextType LinkProgram(
        pro::proxy<VertexShaderFacade> vp,
        pro::proxy<PixelShaderFacade> pp) noexcept;

private:
    glm::vec4 RenderRay(int x, int y);

    CanvasType canvas;
    int width = 0;
    int height = 0;
    glm::mat4 viewMatrix;
    ModelData *modelData = nullptr;
};
