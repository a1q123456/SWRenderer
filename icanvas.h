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
