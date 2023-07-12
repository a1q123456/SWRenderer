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
    colorBufferU8 = CudaNewArray<std::uint8_t>(width * height * 4);
    depthBufferU8 = CudaNewArray<std::uint8_t>(width * height);
    cudaMemset(colorBufferU8.get(), 0, width * height * sizeof(std::uint8_t) * 4);
    cudaMemset(depthBufferU8.get(), 0, width * height * sizeof(std::uint8_t));
}

void RayTracingRenderer::AddMesh(ModelDataType* mesh, ProgramContextType& programCtx)
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
    CudaThrowIfFailed(renderFrame(iprojMatrix, iviewMatrix, width, height, colorBufferU8.get()));
    CudaThrowIfFailed(cudaMemcpy(canvas.Buffer(), colorBufferU8.get(), width * height * 4, cudaMemcpyDeviceToHost));
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
    iprojMatrix = glm::inverse(proj);
}

RayTracingRenderer::ProgramContextType RayTracingRenderer::LinkProgram(PixelProgram* pp) noexcept
{
    return RayTracingRenderer::ProgramContextType{};
}
