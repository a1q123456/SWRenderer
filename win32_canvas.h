#pragma once
#include "tstring.h"
#include "icanvas.h"

class Win32Canvas
{
    int width = 0;
    int height = 0;
    HBITMAP bitmap[2];
    std::uint8_t *buffer[2];
    HDC memCompatiableDc = nullptr;
    int currentBufferIndex = 0;
private:
    void Swap(Win32Canvas& other)
    {
        std::swap(width, other.width);
        std::swap(height, other.height);
        std::swap(bitmap, other.bitmap);
        std::swap(buffer, other.buffer);
        std::swap(memCompatiableDc, other.memCompatiableDc);
    }
public:
    Win32Canvas();
    Win32Canvas(HDC hdc, int w, int h);
    Win32Canvas(const Win32Canvas&) = delete;
    Win32Canvas(Win32Canvas&& other) : Win32Canvas()
    {
        Swap(other);
    }
    Win32Canvas& operator=(const Win32Canvas&) = delete;
    Win32Canvas& operator=(Win32Canvas&& other)
    {
        Swap(other);

        return *this;
    }

    void SwapBuffer() noexcept;
    void Clear(std::uint32_t color) noexcept;
    void LineTo(int x0, int y0, int x1, int y1, std::uint32_t color) noexcept;
    void AddText(int x, int y, int size, const TString& str, std::uint32_t color) noexcept;
    int Width() const noexcept { return width; }
    int Height() const noexcept { return height; };
    std::uint8_t* Buffer() noexcept { return buffer[currentBufferIndex]; }
    HBITMAP Bitmap() const noexcept { return bitmap[currentBufferIndex]; }
    ~Win32Canvas() noexcept;
};
