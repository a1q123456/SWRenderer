#pragma once

struct RendererCanvasDispatchable : pro::dispatch<pro::proxy<CanvasFacade>()>
{
    template <class TSelf>
    pro::proxy<CanvasFacade> operator()(const TSelf& self) const noexcept
    {
        return self.Canvas();
    }
};

struct RendererCreateBufferDispatchable : pro::dispatch<void(EPixelFormat pixelFormat)>
{
    template <class TSelf>
    void operator()(const TSelf& self, EPixelFormat pixelFormat) const noexcept
    {
        return self.CreateBuffer();
    }
};

struct RendererFacade : pro::facade<
    RendererCanvasDispatchable>
{

};
