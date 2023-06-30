#pragma once
#include "tstring.h"

template<typename T>
concept CanvasDrawable = requires(T canvas)
{
    { canvas.SwapBuffer() } noexcept;
    { canvas.Clear(std::declval<std::uint32_t>()) } noexcept;
    { canvas.LineTo(std::declval<int>(), std::declval<int>(), std::declval<int>(), std::declval<int>(), std::declval<std::uint32_t>()) } noexcept;
    { canvas.AddText(std::declval<int>(), std::declval<int>(), std::declval<int>(), std::declval<TString>(), std::declval<std::uint32_t>()) } noexcept;
    { canvas.Buffer() } noexcept -> std::convertible_to<std::uint8_t*>;
    { canvas.Width() } noexcept -> std::same_as<int>;
    { canvas.Height() } noexcept -> std::same_as<int>;
    { std::declval<const T>().Width() } noexcept -> std::same_as<int>;
    { std::declval<const T>().Height() } noexcept -> std::same_as<int>;
};

struct CanvasClearDispatchable : pro::dispatch<void(std::uint32_t color)>
{
    template <class TSelf>
    void operator()(TSelf& self, std::uint32_t color) const noexcept
    {
        return self.Clear(color);
    }
};

struct CanvasLineToDispatchable : pro::dispatch<void(int x0, int y0, int x1, int y1, std::uint32_t color)>
{
    template <class TSelf>
    void operator()(TSelf& self, int x0, int y0, int x1, int y1, std::uint32_t color) const noexcept
    {
        return self.LineTo(x0, y0, x1, y1, color);
    }
};

struct CanvasAddTextDispatchable : pro::dispatch<void(int, int, int, TString, std::uint32_t)>
{
    template <class TSelf>
    void operator()(TSelf& self, int x, int y, int size, const TString &str, std::uint32_t color) const noexcept
    {
        return self.AddText(x, y, size, str, color);
    }
};

struct CanvasBufferDispatchable : pro::dispatch<std::uint8_t*()>
{
    template <class TSelf>
    std::uint8_t* operator()(TSelf& self) const noexcept
    {
        return self.Buffer();
    }
};

struct CanvasHeightDispatchable : pro::dispatch<int()>
{
    template <class TSelf>
    int operator()(const TSelf& self) const noexcept
    {
        return self.Height();
    }
};

struct CanvasWidthDispatchable : pro::dispatch<int()>
{
    template <class TSelf>
    int operator()(const TSelf& self) const noexcept
    {
        return self.Width();
    }
};

struct CanvasSwapBufferDispatchable : pro::dispatch<void()>
{
    template <class TSelf>
    void operator()(TSelf& self) const noexcept
    {
        return self.SwapBuffer();
    }
};


struct CanvasFacade : pro::facade<
    CanvasClearDispatchable,
    CanvasLineToDispatchable,
    CanvasAddTextDispatchable,
    CanvasBufferDispatchable,
    CanvasHeightDispatchable,
    CanvasWidthDispatchable,
    CanvasSwapBufferDispatchable,
    CanvasSwapBufferDispatchable>
{

};