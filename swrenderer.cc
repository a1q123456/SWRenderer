#include "swrenderer.h"
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
#include <glm/vec3.hpp>
#include <glm/common.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#define _USE_MATH_DEFINES
#include <cmath>

SWRenderer::SWRenderer(HDC hdc, int w, int h) : hdc(hdc), width(w), height(h)
{
    memDc = ::CreateCompatibleDC(hdc);
}

void SWRenderer::CreateBuffer(int pixelFormat)
{
    constexpr auto nBuffer = sizeof(buffer) / sizeof(buffer[0]);
    constexpr auto align = (32 / 8);
    bufferLinesize = (width / align) + (align - (width % align));

    BITMAPINFO bm = {sizeof(BITMAPINFOHEADER),
                     width,
                     height, 1, 32,
                     BI_RGB, width * height * 4, 0, 0, 0, 0};

    for (int i = 0; i < nBuffer; i++)
    {
        bitmaps[i] = CreateDIBSection(memDc, &bm, DIB_RGB_COLORS, (void **)&buffer[i], 0, 0);
        canvas[i] = std::make_unique<Canvas>(memDc, bitmaps[i], width, height);
    }
}

void SWRenderer::UpdateBuffer(std::uint8_t *data, int srcWidth, int srcHeight, int linesize)
{
    int idx = !bufferIndex;
    auto h = std::min(height, srcHeight);
    auto w = std::min(width, srcWidth);
    for (auto i = 0; i < h; i++)
    {
        memcpy(buffer[idx] + (i * bufferLinesize), data + (i * linesize), w);
    }
    InterlockedExchange(&bufferIndex, idx);
}

void SWRenderer::SwapBuffer()
{
    InterlockedExchange(&bufferIndex, !bufferIndex);
}

HBITMAP SWRenderer::GetBitmap() const
{
    return bitmaps[bufferIndex];
}

// clang-format off
float vertexList[] = {
    // front
    0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 1, 0,
    1, 1, 0, 1, 1, 0,
    1, 0, 0, 1, 0, 0,

    // left
    0, 0, 1, 0, 0, 0,
    0, 1, 1, 0, 1, 0,
    0, 1, 0, 1, 1, 0,
    0, 0, 0, 1, 0, 0,

    // back
    1, 0, 1, 0, 0, 0,
    1, 1, 1, 0, 1, 0,
    0, 1, 1, 1, 1, 0,
    0, 0, 1, 1, 0, 0,

    // right
    1, 0, 0, 0, 0, 0,
    1, 1, 0, 0, 1, 0,
    1, 1, 1, 1, 1, 0,
    1, 0, 1, 1, 0, 0,

    // bottom
    1, 0, 0, 0, 0, 0,
    1, 0, 1, 0, 1, 0,
    0, 0, 1, 1, 1, 0,
    0, 0, 0, 1, 0, 0,

    // top
    0, 1, 0, 0, 0, 0,
    0, 1, 1, 0, 1, 0,
    1, 1, 1, 1, 1, 0,
    1, 1, 0, 1, 0, 0,
    };

// clang-format on
enum VertexElementType
{
    VET_POSITION,
    VET_UV,
    VET_NORMAL
};

enum DataType
{
    DT_FLOAT3 = sizeof(float) * 3,
    DT_FLOAT2 = sizeof(float) * 2,
    DT_FLOAT1 = sizeof(float) * 1
};

const int vertexLayout[]{
    VET_POSITION, DT_FLOAT3,
    VET_UV, DT_FLOAT3};

int indexList[] = {
    // front
    0, 1, 2,
    0, 2, 3,

    // left
    4, 5, 6,
    4, 6, 7,

    // back
    8, 9, 10,
    8, 10, 11,

    // right
    12, 13, 14,
    12, 14, 15,

    // bottom
    16, 17, 18,
    16, 18, 19,

    // top
    20, 21, 22,
    20, 22, 23};

void SWRenderer::Render(float timeElapsed)
{
    float fov = 50;
    float aspectRatio = (float)width / (float)height;
    float zNear = 0.01;
    float zFar = 1000;

    float canvasWidth = 1;
    float canvasHeight = 1;

    static float x = 0;
    static float y = 0;
    static float z = 0;

    x += timeElapsed * 0.5;
    //y += timeElapsed * 0.1;
    z += timeElapsed * 0.3;

    auto viewTransform = glm::lookAt(glm::vec3{ 0, 0, -7 }, glm::vec3{0, 0, 0}, glm::vec3{0, 1, 0});
    auto projectionMatrix = glm::perspective(glm::radians(55.f), aspectRatio, zNear, zFar);
    auto scaleMatrix = glm::scale(glm::identity<glm::mat4>(), glm::vec3{1, 1, 1});
    auto translateOriginMatrix = glm::translate(glm::identity<glm::mat4>(), glm::vec3{-0.5, -0.5, -0.5});
    auto translateBackMatrix = glm::identity<glm::mat4>();
    auto translateMatrix = glm::translate(glm::identity<glm::mat4>(), glm::vec3{0, 0, 0});
    auto rotationMatrix = glm::eulerAngleXYZ(x, y, z);
    auto modelTransform = translateMatrix * translateBackMatrix * rotationMatrix * scaleMatrix * translateOriginMatrix;
    auto projWorld = projectionMatrix * glm::inverse(viewTransform) * modelTransform;

    canvas[bufferIndex]->Clear(0);
    constexpr int nbIndices = sizeof(indexList) / sizeof(indexList[0]);
    constexpr int nbLayoutElement = sizeof(vertexLayout) / sizeof(vertexLayout[0]);
    int szElement = 0;
    for (int i = 0; i < 4; i += 2)
    {
        szElement += vertexLayout[(i + 1)];
    }
    szElement /= sizeof(float);
    for (int i = 0; i < nbIndices; i += 3)
    {
        glm::vec3 v0;
        glm::vec3 v1;
        glm::vec3 v2;

        glm::vec3 uv0;
        glm::vec3 uv1;
        glm::vec3 uv2;

        int offset = 0;
        for (int j = 0; j < nbLayoutElement; j += 2)
        {
            if (vertexLayout[j] == VET_POSITION)
            {
                v0 = glm::vec3{vertexList[indexList[i + 0] * szElement + offset], vertexList[indexList[i + 0] * szElement + 1 + offset], vertexList[indexList[i + 0] * szElement + 2 + offset]};
                v1 = glm::vec3{vertexList[indexList[i + 1] * szElement + offset], vertexList[indexList[i + 1] * szElement + 1 + offset], vertexList[indexList[i + 1] * szElement + 2 + offset]};
                v2 = glm::vec3{vertexList[indexList[i + 2] * szElement + offset], vertexList[indexList[i + 2] * szElement + 1 + offset], vertexList[indexList[i + 2] * szElement + 2 + offset]};
                offset += vertexLayout[j + 1];
            }
            else if (vertexLayout[j] == VET_UV)
            {
                uv0 = glm::vec3{vertexList[indexList[i + 0] * szElement + offset], vertexList[indexList[i + 0] * szElement + 1 + offset], vertexList[indexList[i + 0] * szElement + 2 + offset]};
                uv1 = glm::vec3{vertexList[indexList[i + 1] * szElement + offset], vertexList[indexList[i + 1] * szElement + 1 + offset], vertexList[indexList[i + 1] * szElement + 2 + offset]};
                uv2 = glm::vec3{vertexList[indexList[i + 2] * szElement + offset], vertexList[indexList[i + 2] * szElement + 1 + offset], vertexList[indexList[i + 2] * szElement + 2 + offset]};
            }
        }

        auto pv0 = projWorld * glm::vec4{v0, 1};
        auto pv1 = projWorld * glm::vec4{v1, 1};
        auto pv2 = projWorld * glm::vec4{v2, 1};

        pv0 = pv0 * projectionMatrix;
        pv1 = pv1 * projectionMatrix;
        pv2 = pv2 * projectionMatrix;

        auto ov0 = pv0;
        auto ov1 = pv1;
        auto ov2 = pv2;

        auto sv0 = glm::vec3{pv0 / -pv0.z};
        auto sv1 = glm::vec3{pv1 / -pv1.z};
        auto sv2 = glm::vec3{pv2 / -pv2.z};

        // backface culling
        auto t0 = sv1 - sv0;
        auto t1 = sv2 - sv1;

        if (glm::cross(t0, t1).z < 0)
        {
            continue;
        }

        // // frustum culling
        // if ()
        // {
        //     continue;
        // }

        sv0 = glm::vec3{(sv0.x + canvasWidth / 2.f) / canvasWidth, (sv0.y + canvasHeight / 2.f) / canvasHeight, ov0.z};
        sv1 = glm::vec3{(sv1.x + canvasWidth / 2.f) / canvasWidth, (sv1.y + canvasHeight / 2.f) / canvasHeight, ov1.z};
        sv2 = glm::vec3{(sv2.x + canvasWidth / 2.f) / canvasWidth, (sv2.y + canvasHeight / 2.f) / canvasHeight, ov2.z};


        glm::vec2 rv0{sv0.x * width, sv0.y * height};
        glm::vec2 rv1{sv1.x * width, sv1.y * height};
        glm::vec2 rv2{sv2.x * width, sv2.y * height};

        canvas[bufferIndex]->LineTo(std::round(rv0.x), std::round(rv0.y), std::round(rv1.x), std::round(rv1.y), 0xFFFFFFFF);
        canvas[bufferIndex]->LineTo(std::round(rv1.x), std::round(rv1.y), std::round(rv2.x), std::round(rv2.y), 0xFFFFFFFF);
        canvas[bufferIndex]->LineTo(std::round(rv2.x), std::round(rv2.y), std::round(rv0.x), std::round(rv0.y), 0xFFFFFFFF);

        // texturing
        
    }
}
