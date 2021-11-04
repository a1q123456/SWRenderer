#pragma once

#include <concepts>
#include "native_window_handle.h"

template<typename T>
concept NativeDisplayable = requires(T window)
{
    { std::declval<T>().Width() } noexcept -> std::same_as<int>;
    { std::declval<T>().Height() } noexcept -> std::same_as<int>;
    { std::declval<T>().GetNativeHandle() } -> std::same_as<NativeWindowHandle>;
    { window.Exec() } -> std::convertible_to<int>;
};