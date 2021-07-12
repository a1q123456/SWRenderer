#pragma once
#include "canvas.h"

class SWRenderer
{
    int width = 500;
    int height = 500;
    HDC hdc;
    HDC memDc;
    std::unique_ptr<Canvas> canvas[2];
    HBITMAP bitmaps[2];
    std::uint8_t *buffer[2];
    volatile long bufferIndex = 0;
    int bufferLinesize = 0;
    void UpdateBuffer(std::uint8_t *data, int srcWidth, int srcHeight, int linesize);

public:
    void CreateBuffer(int pixelFormat);
    SWRenderer(HDC hdc, int w, int h);
    SWRenderer(const SWRenderer&) = delete;
    SWRenderer& operator=(const SWRenderer&) = delete;
    void SwapBuffer();
    void Render(float timeElapsed);
    HBITMAP GetBitmap() const;
};
