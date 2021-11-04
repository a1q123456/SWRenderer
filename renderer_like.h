#pragma once

#include "icanvas.h"

template<typename T>
concept RendererLike = requires(T renderer)
{
    { renderer.ProjectionMatrix(std::declval<glm::mat4>()) };
    { renderer.LinkProgram(std::declval<VertexProgram>(), std::declval<PixelProgram>()) } -> std::same_as<ProgramContext>;
    { renderer.SetMesh(std::declval<ModelData>()) };
    { renderer.SetProgram(std::declval<ProgramContext>()) };
    { renderer.ClearColorBuffer(std::declval<std::uint32_t>()) };
    { renderer.ClearZBuffer() };
    { renderer.Draw(std::declval<float>()) };
    { renderer.Canvas() } -> CanvasDrawable
};
