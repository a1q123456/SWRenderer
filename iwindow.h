#pragma once
#include "native_window_handle.h"

template<typename T>
concept NativeDisplayable = requires(T window)
{
    { std::declval<T>().Width() } noexcept -> std::same_as<int>;
    { std::declval<T>().Height() } noexcept -> std::same_as<int>;
    { std::declval<T>().GetNativeHandle() } -> std::same_as<NativeWindowHandle>;
    { window.Exec() } -> std::convertible_to<int>;
};


struct WindowGetNativeHandleDispatchable : pro::dispatch<NativeWindowHandle()>
{
    template <class TSelf>
    NativeWindowHandle operator()(const TSelf& self) const noexcept
    {
        return self.GetNativeHandle();
    }
};

struct WindowExecDispatchable : pro::dispatch<int()>
{
    template <class TSelf>
    int operator()(const TSelf& self) const noexcept
    {
        return self.Exec();
    }
};

struct WindowWidthDispatchable : pro::dispatch<int()>
{
    template <class TSelf>
    int operator()(const TSelf& self) const noexcept
    {
        return self.Width();
    }
};

struct WindowHeightDispatchable : pro::dispatch<int()>
{
    template <class TSelf>
    int operator()(const TSelf& self) const noexcept
    {
        return self.Height();
    }
};

struct WindowFacade : pro::facade<
    WindowGetNativeHandleDispatchable,
    WindowExecDispatchable,
    WindowWidthDispatchable,
    WindowHeightDispatchable>
{

};
