#include "headless_canvas.h"

HeadlessCanvas::HeadlessCanvas(const HeadlessWindow& window) : 
    width(window.Width()), 
    height(window.Height())
{
    buffer = new std::uint8_t[width * height * 4];
}

void HeadlessCanvas::Clear(std::uint32_t color) noexcept
{
    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            buffer[y * width * sizeof(std::uint32_t) + x * sizeof(std::uint32_t) + 0] = color;
            buffer[y * width * sizeof(std::uint32_t) + x * sizeof(std::uint32_t) + 1] = color >> 8;
            buffer[y * width * sizeof(std::uint32_t) + x * sizeof(std::uint32_t) + 2] = color >> 16;
            buffer[y * width * sizeof(std::uint32_t) + x * sizeof(std::uint32_t) + 3] = color >> 24;
        }
    }
}

void HeadlessCanvas::LineTo(int x0, int y0, int x1, int y1, std::uint32_t color) noexcept
{
    
}

void HeadlessCanvas::AddText(int, int, int, const TString& text, std::uint32_t color) noexcept
{

}

int HeadlessCanvas::Width() const noexcept
{
    return width;
}

int HeadlessCanvas::Height() const noexcept
{
    return height;
}

std::uint8_t* HeadlessCanvas::Buffer() noexcept
{
    return buffer;
}


void HeadlessCanvas::Save(std::filesystem::path dst) const
{
    stbi_flip_vertically_on_write(true);
    stbi_write_png_to_func([](void* context, void *data, int size) {
        auto dst = *reinterpret_cast<std::filesystem::path*>(context);
        std::fstream dstFile{dst, std::ios::out | std::ios::trunc | std::ios::binary};
        dstFile.write((const char*)data, size);
    }, &dst, width, height, STBI_rgb_alpha, buffer, width * 4);
}

void HeadlessCanvas::SwapBuffer() noexcept {}
