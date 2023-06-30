#pragma once
#include "tstring.h"

class HeadlessCanvas
{
public:
    HeadlessCanvas(int w, int h);
    void Clear(std::uint32_t color) noexcept;
    void LineTo(int x0, int y0, int x1, int y1, std::uint32_t color) noexcept;
    void AddText(int, int, int, const TString& text, std::uint32_t color) noexcept;
    int Width() const noexcept;
    int Height() const noexcept;
    std::uint8_t* Buffer() noexcept;
    void Save(std::filesystem::path dst) const;
    void SwapBuffer() noexcept;
private:
    int width = 0;
    int height = 0;
    std::uint8_t* buffer = nullptr;
};