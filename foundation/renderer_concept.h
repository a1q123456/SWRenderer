#pragma once

template<typename T>
concept Renderer = requires(T r)
{
    { r.SetProgram(std::declval<ProgramContext>()); }
    { r.SetMesh(std::declval<ModelData>()); }
    { r.SetCamera(std::declval<Camera>()); }
};
