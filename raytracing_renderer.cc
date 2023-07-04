#include "raytracing_renderer.h"
#include "utils.h"
#include "raytracing_kernels.h"

RayTracingRenderer::RayTracingRenderer(CanvasType&& canvas) : 
    canvas(std::move(canvas)), 
    width(this->canvas.Width()), 
    height(this->canvas.Height())
{
}

void RayTracingRenderer::CreateBuffer(EPixelFormat pixelFormat)
{
    cudaMalloc(reinterpret_cast<void**>(&colorBuffer), width * height * sizeof(float) * 4);
    cudaMalloc(reinterpret_cast<void**>(&depthBuffer), width * height * sizeof(float));
    cudaMemset(colorBuffer, 0, width * height * sizeof(float) * 4);
    cudaMemset(depthBuffer, 0, width * height * sizeof(float));
}

void RayTracingRenderer::SetProgram(RayTracingRenderer::ProgramContextType& programCtx)
{
}

void RayTracingRenderer::SetMesh(RayTracingRenderer::ModelDataType& mesh)
{
    modelData = &mesh;
}

void RayTracingRenderer::ClearZBuffer()
{
}

void RayTracingRenderer::ClearColorBuffer(std::uint32_t color)
{
}

void RayTracingRenderer::Draw(float timeElapsed)
{
    unsigned int numBlocks = 10;
    dim3 threadsPerBlock(width / numBlocks, height / numBlocks);
    dim3 grid{numBlocks, 1, 1};
    cudaLaunchKernel(renderRay, grid, threadsPerBlock, nullptr, 0, nullptr);
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
