#include "raytracing_renderer.h"
#include "utils.h"

RayTracingRenderer::RayTracingRenderer(CanvasType&& canvas) : 
    canvas(std::move(canvas)), 
    width(this->canvas.Width()), 
    height(this->canvas.Height())
{
}

void RayTracingRenderer::CreateBuffer(EPixelFormat pixelFormat)
{
}

void RayTracingRenderer::SetProgram(RayTracingRenderer::ProgramContextType& programCtx)
{
}

void RayTracingRenderer::SetMesh(ModelData& mesh)
{
}

void RayTracingRenderer::ClearZBuffer()
{
}

void RayTracingRenderer::ClearColorBuffer(std::uint32_t color)
{
}

void RayTracingRenderer::Draw(float timeElapsed)
{
}

CanvasType& RayTracingRenderer::Canvas()
{
    return canvas;
}

void RayTracingRenderer::SetMultiSampleLevel(int msaa)
{
}

void RayTracingRenderer::ProjectionMatrix(glm::mat4x4 proj)
{
}

RayTracingRenderer::ProgramContextType RayTracingRenderer::LinkProgram(pro::proxy<VertexShaderFacade> vp,
                                                                        pro::proxy<PixelShaderFacade> pp) noexcept
{
    return RayTracingRenderer::ProgramContextType{};
}

glm::vec4 RayTracingRenderer::RenderRay(int x, int y)
{
    return {};
}
