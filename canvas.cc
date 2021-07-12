#include "canvas.h"

void Canvas::Clear(std::uint32_t color)
{
    auto old = ::SelectObject(hdc, bitmap);
    HRGN rgn = ::CreateRectRgn(0, 0, width, height);
    HBRUSH bsh = ::CreateSolidBrush(color & 0x00FFFFFF);
    ::FillRgn(hdc, rgn, bsh);
    SelectObject(hdc, old);
    DeleteObject(rgn);
    DeleteObject(bsh);
}

void Canvas::LineTo(int x0, int y0, int x1, int y1, std::uint32_t color)
{
    auto old = ::SelectObject(hdc, bitmap);
    HPEN hPen = CreatePen(PS_SOLID, 1, color & 0x00FFFFFF);
    auto oldPen = ::SelectObject(hdc, hPen);
    ::MoveToEx(hdc, x0, y0, nullptr);
    ::LineTo(hdc, x1, y1);
    SelectObject(hdc, oldPen);
    SelectObject(hdc, old);
    DeleteObject(hPen);
}
