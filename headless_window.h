#pragma once

#include "native_window_handle.h"

class HeadlessWindow
{
public:
    HeadlessWindow(int w, int h);
    int Width() const noexcept;
    int Height() const noexcept;
    int Exec();
    NativeWindowHandle GetNativeHandle();
private:
    int width = 0;
    int height = 0;
};
