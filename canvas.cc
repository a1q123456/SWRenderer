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
    ::MoveToEx(hdc, x0, height - y0, nullptr);
    ::LineTo(hdc, x1, height - y1);
    SelectObject(hdc, oldPen);
    SelectObject(hdc, old);
    DeleteObject(hPen);
}

void Canvas::AddText(int x, int y, int size, const TString &str, std::uint32_t color)
{
    auto old = ::SelectObject(hdc, bitmap);
    HPEN hPen = CreatePen(PS_SOLID, 1, color & 0x00FFFFFF);
    auto oldPen = ::SelectObject(hdc, hPen);
    ::MoveToEx(hdc, x, y, nullptr);
    RECT rect;
    ZeroMemory(&rect, sizeof(rect));
    ::DrawText(hdc, str.c_str(), str.size(), &rect, DT_CALCRECT);
    ::DrawText(hdc, str.c_str(), str.size(), &rect, DT_LEFT);
    SelectObject(hdc, oldPen);
    SelectObject(hdc, old);
    DeleteObject(hPen);
}
