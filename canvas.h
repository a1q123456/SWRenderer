#pragma once
#include "vec.h"


class Canvas
{
    int width;
    int height;
    HBITMAP bitmap;
    HDC hdc;
public:
    Canvas(HDC hdc, HBITMAP bmp, int w, int h) : hdc(hdc), bitmap(bmp), width(w), height(h) { }
    void Clear(std::uint32_t color);
    void LineTo(int x0, int y0, int x1, int y1, std::uint32_t color);
};
