#pragma once

#include "icanvas.h"
#include "swrenderer.h"
#include "shading/material/simple_vertex_program.h"
#include "shading/material/blinn_material.h"
#include "native_window_handle.h"
#include "pixel_format.h"
#include "texture.h"
#include <iostream>


template<CanvasDrawable TCanvas>
class TestSceneController
{
    int width = 0;
    int height = 0;

    std::shared_ptr<std::uint8_t> textureData;
    int textureW = 0;
    int textureH = 0;
    int textureChannels = 0;
    NativeWindowHandle hwnd = 0;
    bool mouseCaptured = false;
    int lastMouseX = -1;
    int lastMouseY = -1;
    TCanvas canvas;
    Texture2D texture;

public:
    template<CanvasDrawable T>
    TestSceneController(T&& canvas) : canvas(std::forward<T>(canvas))
    {
        width = this->canvas.Width();
        height = this->canvas.Height();
        textureData.reset(stbi_load("Lenna_small.png", &textureW, &textureH, &textureChannels, STBI_default), stbi_image_free);
        texture = Texture2D{
            { std::span{textureData.get(), static_cast<std::size_t>(textureW * textureH * textureChannels)} }, 
            TextureDesc2D{
                1, Texture2DBoundary{textureW, textureH}, textureW * textureChannels, EResourceDataType::UInt8, textureChannels == 3 ? EPixelFormat::RGB_U8 : EPixelFormat::RGBA_U8
            },
            TextureFilteringMethods::Linear,
            0
        };

        auto start = std::chrono::steady_clock::now();
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                auto color = texture.Sample<float, 4>({static_cast<float>(x) / width, 1.0 - static_cast<float>(y) / height});
                // glm::vec4 color{255 * (double)y / height, 255 * (double)x / width, 0, 0};
                this->canvas.Buffer()[y * width * 4 + x * 4 + 0] = std::clamp<float>(color.b, 0, 255);
                this->canvas.Buffer()[y * width * 4 + x * 4 + 1] = std::clamp<float>(color.g, 0, 255);
                this->canvas.Buffer()[y * width * 4 + x * 4 + 2] = std::clamp<float>(color.r, 0, 255);
            }
        }
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        std::stringstream ss;
        ss << ms << "ms" << std::endl;
        OutputDebugStringA(ss.str().c_str());
    }
    
    void CreateBuffer(EPixelFormat pixelFormat) { }
    void SetHWND(NativeWindowHandle hwnd) { this->hwnd = hwnd; }

    void MouseDown() {}

    void MouseUp() {}

    void MouseWheel(int val) {}

    void MouseMove(int x, int y) {}

    void Render(float timeElapsed) 
    {
        // for (int y = 0; y < height; y++)
        // {
        //     for (int x = 0; x < width; x++)
        //     {
        //         auto color = texture.Sample<float, 4>({static_cast<float>(x) / width, 1.0 - static_cast<float>(y) / height});
        //         // glm::vec4 color{255 * (double)y / height, 255 * (double)x / width, 0, 0};
        //         canvas.Buffer()[y * width * 4 + x * 4 + 0] = color.b;
        //         canvas.Buffer()[y * width * 4 + x * 4 + 1] = color.g;
        //         canvas.Buffer()[y * width * 4 + x * 4 + 2] = color.r;
        //     }
        // }
    }

    auto& Canvas() noexcept
    {
        return canvas;
    }
};
