#include "win32_canvas.h"
#include "utils.h"

Win32Canvas::Win32Canvas()
{
    memset(bitmap, 0, sizeof(bitmap));
    memset(buffer, 0, sizeof(buffer));
}

Win32Canvas::Win32Canvas(HDC hdc, int w, int h): width(w), height(h)
{
    memCompatiableDc = CreateCompatibleDC(hdc);
    BITMAPINFO bm = {sizeof(BITMAPINFOHEADER),
                     width,
                     height, 1, 32,
                     BI_RGB, static_cast<DWORD>(width * height * 4), 0, 0, 0, 0};
    for (int i = 0; i < count_of(bitmap); i++)
    {
        bitmap[i] = CreateDIBSection(memCompatiableDc, &bm, DIB_RGB_COLORS, (void **)&buffer[i], 0, 0);
    }
}

Win32Canvas::~Win32Canvas() noexcept
{
    if (memCompatiableDc)
    {
        DeleteDC(memCompatiableDc);

        for (int i = 0; i < count_of(bitmap); i++)
        {
            DeleteObject(bitmap[i]);
        }
    }
}


void Win32Canvas::SwapBuffer() noexcept
{
    currentBufferIndex++;
    if (currentBufferIndex == count_of(bitmap))
    {
        currentBufferIndex = 0;
    }
}

void Win32Canvas::Clear(std::uint32_t color) noexcept
{
    auto old = ::SelectObject(memCompatiableDc, bitmap[currentBufferIndex]);
    HRGN rgn = ::CreateRectRgn(0, 0, width, height);
    HBRUSH bsh = ::CreateSolidBrush(color & 0x00FFFFFF);
    ::FillRgn(memCompatiableDc, rgn, bsh);
    SelectObject(memCompatiableDc, old);
    DeleteObject(rgn);
    DeleteObject(bsh);
}

void Win32Canvas::LineTo(int x0, int y0, int x1, int y1, std::uint32_t color) noexcept
{
    auto old = ::SelectObject(memCompatiableDc, bitmap[currentBufferIndex]);
    HPEN hPen = CreatePen(PS_SOLID, 1, color & 0x00FFFFFF);
    auto oldPen = ::SelectObject(memCompatiableDc, hPen);
    ::MoveToEx(memCompatiableDc, x0, y0, nullptr);
    ::LineTo(memCompatiableDc, x1, y1);
    SelectObject(memCompatiableDc, oldPen);
    SelectObject(memCompatiableDc, old);
    DeleteObject(hPen);
}

void Win32Canvas::AddText(int x, int y, int size, const TString &str, std::uint32_t color) noexcept
{
    auto old = ::SelectObject(memCompatiableDc, bitmap[currentBufferIndex]);
    HPEN hPen = CreatePen(PS_SOLID, 1, color & 0x00FFFFFF);
    auto oldPen = ::SelectObject(memCompatiableDc, hPen);
    ::MoveToEx(memCompatiableDc, x, y, nullptr);
    RECT rect;
    ZeroMemory(&rect, sizeof(rect));
    ::DrawText(memCompatiableDc, str.c_str(), str.size(), &rect, DT_CALCRECT);
    ::DrawText(memCompatiableDc, str.c_str(), str.size(), &rect, DT_LEFT);
    SelectObject(memCompatiableDc, oldPen);
    SelectObject(memCompatiableDc, old);
    DeleteObject(hPen);
}
