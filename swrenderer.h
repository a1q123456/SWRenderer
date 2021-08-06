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
    long bufferIndex = 0;
    float* zBuffer = nullptr;
    bool depthTestEnabled = true;
    bool depthWriteEnabled = true;
    bool backFaceCulling = true;
    
    std::uint8_t* textureData = nullptr;
    int textureW = 0;
    int textureH = 0;
    int textureChannels = 0;

    void ClearZBuffer();

public:
    void CreateBuffer(int pixelFormat);
    SWRenderer(HDC hdc, int w, int h);
    SWRenderer(const SWRenderer&) = delete;
    SWRenderer& operator=(const SWRenderer&) = delete;
    void SwapBuffer();
    void Render(float timeElapsed);
    HBITMAP GetBitmap() const;
};
